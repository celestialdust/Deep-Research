from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, get_buffer_string, filter_messages
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command, interrupt
import asyncio
import os
from datetime import datetime
from typing import Literal
from .configuration import (
    Configuration, 
)
from .state import (
    AgentState,
    AgentInputState,
    SupervisorState,
    ResearcherState,
    ClarifyWithUser,
    ResearchQuestion,
    DraftReport,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState
)
from .prompts import (
    clarify_with_user_instructions,
    transform_messages_into_research_topic_prompt,
    draft_report_generation_prompt,
    research_system_prompt,
    compress_research_system_prompt,
    compress_research_simple_human_message,
    final_report_generation_prompt,
    lead_researcher_prompt,
    citation_check_prompt,
    convert_refs_to_urls_prompt
)
from .utils import (
    get_today_str,
    is_token_limit_exceeded,
    get_model_token_limit,
    get_all_tools,
    openai_websearch_called,
    anthropic_websearch_called,
    remove_up_to_last_ai_message,
    get_api_key_for_model,
    get_notes_from_tool_calls,
    build_model_config,
    think_tool,
    citation_think_tool,
    refine_draft_report,
    tavily_search,
    MessageSummarizer,
    invoke_model_with_summarization,
    extract_title_from_markdown,
    sanitize_filename,
    generate_pdf_from_markdown,
    save_markdown_file
)

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key", "model_provider", "azure_endpoint", "azure_deployment", "api_version"),
)

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        return Command(goto="write_research_brief")
    messages = state["messages"]
    model_config = build_model_config(
        configurable.research_model,
        configurable.research_model_max_tokens,
        config,
        ["langsmith:nostream"]
    )
    model = configurable_model.with_structured_output(ClarifyWithUser).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(model_config)
    response = await model.ainvoke([HumanMessage(content=clarify_with_user_instructions.format(messages=get_buffer_string(messages), date=get_today_str()))])
    if response.need_clarification:
        return Command(goto=END, update={"messages": [AIMessage(content=response.question)]})
    else:
        return Command(goto="write_research_brief", update={"messages": [AIMessage(content=response.verification)]})


async def write_research_brief(state: AgentState, config: RunnableConfig)-> Command[Literal["approve_research_brief"]]:
    configurable = Configuration.from_runnable_config(config)
    research_model_config = build_model_config(
        configurable.research_model,
        configurable.research_model_max_tokens,
        config,
        ["langsmith:nostream"]
    )
    
    # Check if search is enabled for brief generation
    raw_notes_content = None
    if configurable.enable_search_for_brief:
        # Use tool-calling model with Tavily search (max 1 search allowed)
        research_model_with_tools = configurable_model.bind_tools([tavily_search]).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(research_model_config)
        
        # Initial call - may include tool calls
        initial_messages = [HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))]
        initial_response = await research_model_with_tools.ainvoke(initial_messages)
        
        # If there are tool calls, execute them (max 1 iteration)
        if initial_response.tool_calls:
            tool_call = initial_response.tool_calls[0]  # Only execute first tool call
            tool_result = await tavily_search.ainvoke(tool_call["args"], config)
            
            # Format raw notes like research agents do
            raw_notes_content = "\n".join([
                str(initial_response.content) if initial_response.content else "",
                str(tool_result)
            ])
            
            # Add tool message and get structured output
            messages_with_tool = initial_messages + [
                initial_response,
                ToolMessage(content=tool_result, name=tool_call["name"], tool_call_id=tool_call["id"])
            ]
            
            # Now get structured output
            research_model_structured = configurable_model.with_structured_output(ResearchQuestion).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(research_model_config)
            response = await research_model_structured.ainvoke(messages_with_tool)
        else:
            # No tool calls, get structured output directly
            research_model_structured = configurable_model.with_structured_output(ResearchQuestion).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(research_model_config)
            response = await research_model_structured.ainvoke(initial_messages)
    else:
        # No search - use structured output directly
        research_model = configurable_model.with_structured_output(ResearchQuestion).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(research_model_config)
        response = await research_model.ainvoke([HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))])
    
    update_dict = {
        "research_brief": response.research_brief
    }
    if raw_notes_content:
        update_dict["raw_notes"] = [raw_notes_content]
    
    return Command(
        goto="approve_research_brief", 
        update=update_dict
    )


async def approve_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "write_draft_report"]]:
    """
    Human-in-the-Loop checkpoint for research brief approval.
    
    This node pauses execution and waits for human approval or rejection of the research brief.
    Based on LangGraph interrupt patterns from documentation.
    
    Decision format (resume value):
        {"type": "approve"} - Continue to write_draft_report
        {"type": "reject", "feedback": "..."} - Loop back to write_research_brief
    
    If HITL is disabled, proceeds directly to write_draft_report.
    
    Args:
        state: Current agent state containing research_brief
        config: Runnable configuration
        
    Returns:
        Command to route to either write_draft_report (approved) or write_research_brief (rejected)
    """
    configurable = Configuration.from_runnable_config(config)
    
    # If HITL is disabled, proceed directly to draft report
    if not configurable.enable_human_in_the_loop:
        return Command(goto="write_draft_report")
    
    # Get current refinement round and research brief
    brief_refinement_rounds = state.get("brief_refinement_rounds", 0)
    research_brief = state.get("research_brief")
    
    if not research_brief:
        return Command(goto="write_draft_report")
    
    # Prepare interrupt payload following LangGraph HITL patterns
    interrupt_payload = {
        "research_brief": research_brief,
        "refinement_round": brief_refinement_rounds,
        "max_rounds": configurable.max_brief_refinement_rounds,
        "instructions": "Review the research brief. Respond with: {\"type\": \"approve\"} or {\"type\": \"reject\", \"feedback\": \"your feedback here\"}"
    }
    
    # Call interrupt() - this pauses execution and returns the resume value
    decision = interrupt(interrupt_payload)
    
    # Process the decision
    # Expected format: {"type": "approve"} or {"type": "reject", "feedback": "..."}
    if not isinstance(decision, dict):
        return Command(goto="write_draft_report")
    
    decision_type = decision.get("type")
    
    if decision_type == "approve":
        # Approved - proceed to write_draft_report
        return Command(goto="write_draft_report")
        
    elif decision_type == "reject":
        # Rejected - check if we can refine again
        feedback = decision.get("feedback", "Please revise the research brief to better address the requirements.")
        
        if brief_refinement_rounds >= configurable.max_brief_refinement_rounds:
            # Max rounds reached - proceed anyway
            return Command(goto="write_draft_report")
        else:
            # Loop back to write_research_brief with feedback
            return Command(
                goto="write_research_brief",
                update={
                    "messages": [HumanMessage(content=f"The research brief was rejected. Please revise based on this feedback:\n\n{feedback}")],
                    "brief_refinement_rounds": brief_refinement_rounds + 1
                }
            )
    else:
        # Unknown decision type - treat as approval for safety
        return Command(goto="write_draft_report")


async def write_draft_report(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Generate initial draft report from research brief.
    
    Creates a preliminary draft report that will be iteratively refined
    through the research process using the diffusion algorithm.
    
    Args:
        state: Current agent state with research brief
        config: Runtime configuration
        
    Returns:
        Command to proceed to research supervisor with draft report
    """
    configurable = Configuration.from_runnable_config(config)
    draft_model_config = build_model_config(
        configurable.research_model,
        configurable.research_model_max_tokens,
        config,
        ["langsmith:nostream"]
    )
    
    # Check if search is enabled for draft generation
    raw_notes_content = None
    if configurable.enable_search_for_draft:
        # Use tool-calling model with Tavily search (max 1 search allowed)
        draft_model_with_tools = configurable_model.bind_tools([tavily_search]).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(draft_model_config)
        
        # Initial call - may include tool calls
        initial_messages = [HumanMessage(content=draft_report_generation_prompt.format(
            research_brief=state.get("research_brief", ""),
            date=get_today_str()
        ))]
        initial_response = await draft_model_with_tools.ainvoke(initial_messages)
        
        # If there are tool calls, execute them (max 1 iteration)
        if initial_response.tool_calls:
            # Execute only the first tool call for search, but create responses for ALL
            tool_results = []
            raw_notes_parts = [str(initial_response.content) if initial_response.content else ""]
            
            for i, tool_call in enumerate(initial_response.tool_calls):
                if i == 0:
                    # Actually execute the first search
                    tool_result = await tavily_search.ainvoke(tool_call["args"], config)
                    raw_notes_parts.append(str(tool_result))
                else:
                    # For additional tool calls, provide a placeholder response
                    # This satisfies OpenAI's requirement that all tool_calls have responses
                    tool_result = "Search skipped - only one search allowed per draft generation."
                
                tool_results.append(
                    ToolMessage(content=tool_result, name=tool_call["name"], tool_call_id=tool_call["id"])
                )
            
            # Format raw notes
            raw_notes_content = "\n".join(raw_notes_parts)
            
            # Add all tool messages to satisfy OpenAI API requirement
            messages_with_tool = initial_messages + [initial_response] + tool_results
            
            # Now get structured output
            draft_model_structured = configurable_model.with_structured_output(DraftReport).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(draft_model_config)
            response = await draft_model_structured.ainvoke(messages_with_tool)
        else:
            # No tool calls, get structured output directly
            draft_model_structured = configurable_model.with_structured_output(DraftReport).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(draft_model_config)
            response = await draft_model_structured.ainvoke(initial_messages)
    else:
        # No search - use structured output directly
        draft_model = configurable_model.with_structured_output(DraftReport).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(draft_model_config)
        response = await draft_model.ainvoke([HumanMessage(content=draft_report_generation_prompt.format(
            research_brief=state.get("research_brief", ""),
            date=get_today_str()
        ))])
    
    update_dict = {
        "draft_report": response.draft_report,
        "supervisor_messages": {
            "type": "override",
            "value": [
                SystemMessage(content=lead_researcher_prompt.format(
                    date=get_today_str(),
                    max_concurrent_research_units=configurable.max_concurrent_research_units,
                    max_researcher_iterations=configurable.max_researcher_iterations
                )),
                HumanMessage(content=state.get("research_brief", ""))
            ]
        }
    }
    if raw_notes_content:
        update_dict["raw_notes"] = [raw_notes_content]
    
    return Command(
        goto="research_supervisor",
        update=update_dict
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    configurable = Configuration.from_runnable_config(config)
    research_model_config = build_model_config(
        configurable.research_model,
        configurable.research_model_max_tokens,
        config,
        ["langsmith:nostream"]
    )
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool, refine_draft_report]
    research_model = configurable_model.bind_tools(lead_researcher_tools).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(research_model_config)
    supervisor_messages = state.get("supervisor_messages", [])
    
    response = await invoke_model_with_summarization(
        model=research_model,
        messages=supervisor_messages,
        config=config,
        message_key="supervisor_messages"
    )
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )


async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    
    # Exit Criteria
    # 1. We have exceeded our max guardrail research iterations
    # 2. No tool calls were made by the supervisor
    # 3. The most recent message contains a ResearchComplete tool call
    exceeded_allowed_iterations = research_iterations >= configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(tool_call["name"] == "ResearchComplete" for tool_call in most_recent_message.tool_calls)
    
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", ""),
                "draft_report": state.get("draft_report", "")
            }
        )
    
    # Otherwise, execute tool calls and gather results
    try:
        tool_messages = []
        all_raw_notes = []
        updated_draft_report = state.get("draft_report", "")
        
        # Separate tool calls by type
        think_tool_calls = [tc for tc in most_recent_message.tool_calls if tc["name"] == "think_tool"]
        conduct_research_calls = [tc for tc in most_recent_message.tool_calls if tc["name"] == "ConductResearch"]
        refine_report_calls = [tc for tc in most_recent_message.tool_calls if tc["name"] == "refine_draft_report"]
        
        # Handle think_tool calls
        for tool_call in think_tool_calls:
            observation = think_tool.invoke(tool_call["args"])
            tool_messages.append(ToolMessage(
                content=observation,
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            ))
        
        # Handle ConductResearch calls
        if conduct_research_calls:
            all_conduct_research_calls = conduct_research_calls
            conduct_research_calls = all_conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = all_conduct_research_calls[configurable.max_concurrent_research_units:]
            
            researcher_system_prompt = research_system_prompt.format(mcp_prompt=configurable.mcp_prompt or "", date=get_today_str())
            coros = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        SystemMessage(content=researcher_system_prompt),
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"]
                }, config) 
                for tool_call in conduct_research_calls
            ]
            tool_results = await asyncio.gather(*coros)
            
            research_tool_messages = [ToolMessage(
                content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            ) for observation, tool_call in zip(tool_results, conduct_research_calls)]
            
            tool_messages.extend(research_tool_messages)
            
            # Handle overflow
            for overflow_call in overflow_conduct_research_calls:
                tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))
            
            raw_notes_concat = "\n".join(["\n".join(observation.get("raw_notes", [])) for observation in tool_results])
            all_raw_notes.append(raw_notes_concat)
        
        # Handle refine_draft_report calls
        for tool_call in refine_report_calls:
            notes = get_notes_from_tool_calls(supervisor_messages)
            findings = "\n".join(notes)
            
            draft_report = refine_draft_report.invoke({
                "research_brief": state.get("research_brief", ""),
                "findings": findings,
                "draft_report": state.get("draft_report", "")
            }, config)
            
            updated_draft_report = draft_report
            tool_messages.append(ToolMessage(
                content=draft_report,
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            ))
        
        return Command(
            goto="supervisor",
            update={
                "supervisor_messages": tool_messages,
                "raw_notes": all_raw_notes if all_raw_notes else [],
                "draft_report": updated_draft_report
            }
        )
        
    except Exception as e:
        if is_token_limit_exceeded(e, configurable.research_model):
            print(f"Token limit exceeded while reflecting: {e}")
        else:
            print(f"Other error in reflection phase: {e}")
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", ""),
                "draft_report": state.get("draft_report", "")
            }
        )


supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_subgraph = supervisor_builder.compile()


async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError("No tools found to conduct research: Please configure either your search API or add MCP tools to your configuration.")
    research_model_config = build_model_config(
        configurable.research_model,
        configurable.research_model_max_tokens,
        config,
        ["langsmith:nostream"]
    )
    research_model = configurable_model.bind_tools(tools).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(research_model_config)
    
    response = await invoke_model_with_summarization(
        model=research_model,
        messages=researcher_messages,
        config=config,
        message_key="researcher_messages"
    )
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )


async def execute_tool_safely(tool, args, config):
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    # Early Exit Criteria: No tool calls (or native web search calls)were made by the researcher
    if not most_recent_message.tool_calls and not (openai_websearch_called(most_recent_message) or anthropic_websearch_called(most_recent_message)):
        return Command(
            goto="compress_research",
        )
    # Otherwise, execute tools and gather results.
    tools = await get_all_tools(config)
    tools_by_name = {tool.name if hasattr(tool, "name") else tool.get("name", "web_search"):tool for tool in tools}
    tool_calls = most_recent_message.tool_calls
    coros = [execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config) for tool_call in tool_calls]
    observations = await asyncio.gather(*coros)
    tool_outputs = [ToolMessage(
                        content=observation,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    ) for observation, tool_call in zip(observations, tool_calls)]
    
    # Late Exit Criteria: We have exceeded our max guardrail tool call iterations or the most recent message contains a ResearchComplete tool call
    # These are late exit criteria because we need to add ToolMessages
    if state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls or any(tool_call["name"] == "ResearchComplete" for tool_call in most_recent_message.tool_calls):
        return Command(
            goto="compress_research",
            update={
                "researcher_messages": tool_outputs,
            }
        )
    return Command(
        goto="researcher",
        update={
            "researcher_messages": tool_outputs,
        }
    )


async def compress_research(state: ResearcherState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    synthesis_attempts = 0
    synthesizer_model = configurable_model.with_config(build_model_config(
        configurable.compression_model,
        configurable.compression_model_max_tokens,
        config,
        ["langsmith:nostream"]
    ))
    researcher_messages = state.get("researcher_messages", [])
    research_topic = state.get("research_topic", "")
    # Update the system prompt to now focus on compression rather than research.
    researcher_messages[0] = SystemMessage(content=compress_research_system_prompt.format(date=get_today_str()))
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message.format(research_topic=research_topic)))
    
    while synthesis_attempts < 3:
        try:
            response = await invoke_model_with_summarization(
                model=synthesizer_model,
                messages=researcher_messages,
                config=config,
                message_key="researcher_messages"
            )
            return {
                "compressed_research": str(response.content),
                "raw_notes": ["\n".join([str(m.content) for m in filter_messages(researcher_messages, include_types=["tool", "ai"])])]
            }
        except Exception as e:
            synthesis_attempts += 1
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                print(f"Token limit exceeded while synthesizing: {e}. Pruning the messages to try again.")
                continue         
            print(f"Error synthesizing research report: {e}")
    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": ["\n".join([str(m.content) for m in filter_messages(researcher_messages, include_types=["tool", "ai"])])]
    }


researcher_builder = StateGraph(ResearcherState, output=ResearcherOutputState, config_schema=Configuration)
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("compress_research", compress_research)
researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("compress_research", END)
researcher_subgraph = researcher_builder.compile()


async def final_report_generation(state: AgentState, config: RunnableConfig):
    notes = state.get("notes", [])
    raw_notes = state.get("raw_notes", [])
    cleared_state = {"notes": {"type": "override", "value": []},}
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = build_model_config(
        configurable.final_report_model,
        configurable.final_report_model_max_tokens,
        config
    )
    
    # Helper function to save notes AFTER successful report generation
    # This prevents duplicate saves when LangGraph retries the node
    async def save_notes_async():
        """Save notes to files using async operations to avoid blocking."""
        def write_file(path: str, content: str):
            """Helper to write file with proper resource management."""
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        docs_dir = os.path.join(os.path.dirname(__file__), '../../docs')
        await asyncio.to_thread(os.makedirs, docs_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save notes
        if notes:
            notes_path = os.path.join(docs_dir, f'notes_{timestamp}.md')
            content = f"# Research Notes - {timestamp}\n\n" + '\n\n---\n\n'.join(notes)
            await asyncio.to_thread(write_file, notes_path, content)
            print(f"Notes saved to: {notes_path}")
        
        # Save raw_notes
        if raw_notes:
            raw_notes_path = os.path.join(docs_dir, f'raw_notes_{timestamp}.md')
            content = f"# Raw Research Notes - {timestamp}\n\n" + '\n\n---\n\n'.join(raw_notes)
            await asyncio.to_thread(write_file, raw_notes_path, content)
            print(f"Raw notes saved to: {raw_notes_path}")
    
    findings = "\n".join(notes)
    max_retries = 3
    current_retry = 0
    while current_retry <= max_retries:
        final_report_prompt = final_report_generation_prompt.format(
            research_brief=state.get("research_brief", ""),
            findings=findings,
            draft_report=state.get("draft_report", ""),
            date=get_today_str()
        )
        try:
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([HumanMessage(content=final_report_prompt)])
            
            # Save notes AFTER successful report generation (prevents duplicates on retry)
            try:
                await save_notes_async()
            except Exception as e:
                print(f"Warning: Failed to save notes to /docs: {e}")
            
            return {
                "final_report": final_report.content, 
                "messages": [final_report],
                **cleared_state
            }
        except Exception as e:
            if is_token_limit_exceeded(e, configurable.final_report_model):
                if current_retry == 0:
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            **cleared_state
                        }
                    findings_token_limit = model_token_limit * 4
                else:
                    findings_token_limit = int(findings_token_limit * 0.9)
                print("Reducing the chars to", findings_token_limit)
                findings = findings[:findings_token_limit]
                current_retry += 1
            else:
                # If not a token limit exceeded error, then we just throw an error.
                return {
                    "final_report": f"Error generating final report: {e}",
                    **cleared_state
                }
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [final_report],
        **cleared_state
    }


async def check_citations(state: AgentState, config: RunnableConfig):
    """Verify that all <ref> citations match research notes exactly.
    
    Uses citation_think_tool to analyze each citation and correct any
    paraphrased or hallucinated source sentences.
    
    Args:
        state: Current agent state containing final_report and notes
        config: Runtime configuration
        
    Returns:
        Dictionary with corrected final_report
    """
    from langchain.agents import create_agent
    
    configurable = Configuration.from_runnable_config(config)
    
    # Build model config for citation check
    citation_model_config = build_model_config(
        configurable.citation_check_model,
        configurable.citation_check_model_max_tokens,
        config
    )
    
    final_report = state.get("final_report", "")
    notes = state.get("notes", [])
    
    # If no report or no notes, skip checking
    if not final_report or not notes:
        return {"final_report": final_report}
    
    # Combine notes for context
    notes_combined = "\n\n---\n\n".join(notes)
    
    # Format the prompt
    check_prompt = citation_check_prompt.format(
        notes=notes_combined,
        report=final_report
    )
    
    # Create agent with citation_think_tool using new LangChain v1 API
    citation_agent = create_agent(
        model=configurable_model.with_config(citation_model_config),
        tools=[citation_think_tool],
        system_prompt=check_prompt
    )
    
    try:
        # Run the citation check agent
        result = await citation_agent.ainvoke({"messages": [HumanMessage(content="Please verify and correct all citations in the report.")]})
        
        # Extract the corrected report from the last AI message
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                corrected_report = msg.content
                print("Citation check completed successfully")
                return {"final_report": corrected_report}
        
        # If no corrected report found, return original
        print("Citation check: No corrections made")
        return {"final_report": final_report}
        
    except Exception as e:
        print(f"Citation check failed: {e}")
        # On error, return original report
        return {"final_report": final_report}


async def convert_refs_to_urls(state: AgentState, config: RunnableConfig):
    """Convert <ref> tags to text-fragment URLs for PDF generation.
    
    Uses an LLM to convert <ref id="N">"source sentence"</ref> tags
    to markdown links with text-fragment URLs in range format.
    
    Args:
        state: Current agent state containing final_report
        config: Runtime configuration
        
    Returns:
        Dictionary with final_report_pdf containing URL-converted content string
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Build model config for PDF conversion
    conversion_model_config = build_model_config(
        configurable.pdf_conversion_model,
        configurable.pdf_conversion_model_max_tokens,
        config
    )
    
    final_report = state.get("final_report", "")
    
    if not final_report:
        return {"final_report_pdf": ""}
    
    # Format the conversion prompt
    conversion_prompt = convert_refs_to_urls_prompt.format(report=final_report)
    
    try:
        # Invoke the model to convert refs to URLs
        result = await configurable_model.with_config(conversion_model_config).ainvoke([
            HumanMessage(content=conversion_prompt)
        ])
        
        converted_report = result.content if result.content else final_report
        print("Ref to URL conversion completed successfully")
        return {"final_report_pdf": converted_report}
        
    except Exception as e:
        print(f"Ref to URL conversion failed: {e}")
        # On error, use original report (refs won't be clickable but content preserved)
        return {"final_report_pdf": final_report}


async def convert_to_pdf(state: AgentState, config: RunnableConfig):
    """Convert the final report from markdown to PDF.
    
    Uses WeasyPrint to convert markdown → HTML → PDF with professional styling.
    Uses final_report_pdf (with text-fragment URLs) for PDF generation.
    Adds markdown report to messages for UI display.
    
    Args:
        state: Current agent state containing final_report and final_report_pdf
        config: Runtime configuration
        
    Returns:
        Dictionary with pdf_path, md_path, and messages
    """
    final_report = state.get("final_report", "")
    # Use URL-converted version for PDF, fall back to original
    pdf_ready_content = state.get("final_report_pdf", "") or final_report
    
    if not final_report:
        return {
            "pdf_path": None
        }
    
    # Extract and sanitize title for filename
    title = extract_title_from_markdown(final_report)
    sanitized_title = sanitize_filename(title)
    
    # Setup paths
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    docs_dir = os.path.join(os.path.dirname(__file__), '../../docs')
    
    # Ensure docs directory exists
    try:
        await asyncio.to_thread(os.makedirs, docs_dir, exist_ok=True)
    except Exception as e:
        print(f"Warning: Failed to create docs directory: {e}")
        return {
            "pdf_path": None,
            "messages": [AIMessage(content=final_report)]
        }
    
    pdf_path = os.path.join(docs_dir, f"{sanitized_title}_{timestamp}.pdf")
    md_path = os.path.join(docs_dir, f"{sanitized_title}_{timestamp}.md")
    
    # Save markdown version with text-fragment URLs for reference
    md_success = await save_markdown_file(pdf_ready_content, md_path)
    if md_success:
        print(f"Markdown saved to: {md_path}")
    else:
        print(f"Warning: Failed to save markdown file")
    
    # Generate PDF from URL-converted content
    pdf_success = await generate_pdf_from_markdown(pdf_ready_content, pdf_path, title)
    
    # Create the final message with markdown report (appears in UI)
    final_message = AIMessage(content=final_report)
    
    if pdf_success:
        print(f"PDF generated successfully: {pdf_path}")
        return {
            "pdf_path": pdf_path,
            "md_path": md_path,
            "messages": [final_message]
        }
    
    # PDF generation failed - markdown already saved above
    if md_success:
        print(f"PDF generation failed, markdown available at: {md_path}")
        return {
            "pdf_path": None,
            "md_path": md_path,
            "messages": [final_message]
        }
    
    return {
        "pdf_path": None,
        "md_path": None,
        "messages": [final_message]
    }


deep_researcher_builder = StateGraph(AgentState, input=AgentInputState, config_schema=Configuration)
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("approve_research_brief", approve_research_brief)
deep_researcher_builder.add_node("write_draft_report", write_draft_report)
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)
deep_researcher_builder.add_node("check_citations", check_citations)
deep_researcher_builder.add_node("convert_refs_to_urls", convert_refs_to_urls)
deep_researcher_builder.add_node("convert_to_pdf", convert_to_pdf)
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", "check_citations")
deep_researcher_builder.add_edge("check_citations", "convert_refs_to_urls")
deep_researcher_builder.add_edge("convert_refs_to_urls", "convert_to_pdf")
deep_researcher_builder.add_edge("convert_to_pdf", END)

graph = deep_researcher_builder.compile(name = "deep_researcher")