import os
import uuid
import aiohttp
import asyncio
import logging
import warnings
from datetime import datetime, timedelta, timezone
from typing import Annotated, List, Literal, Dict, Optional, Any, cast, Callable, Iterable
from langchain_core.tools import BaseTool, StructuredTool, tool, ToolException, InjectedToolArg
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AnyMessage, MessageLikeRepresentation, filter_messages, RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages

TokenCounter = Callable[[Iterable[MessageLikeRepresentation]], int]
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseChatModel
from langchain.chat_models import init_chat_model
from tavily import AsyncTavilyClient
from langgraph.config import get_store
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from mcp import McpError
from langchain_mcp_adapters.client import MultiServerMCPClient
from .state import Summary, ResearchComplete
from .configuration import SearchAPI, Configuration
from .prompts import summarize_webpage_prompt, report_generation_with_draft_insight_prompt, context_summarization_prompt


##########################
# Context Summarization
##########################

_DEFAULT_MESSAGES_TO_KEEP = 20
_DEFAULT_TRIM_TOKEN_LIMIT = 4000
_DEFAULT_FALLBACK_MESSAGE_COUNT = 15
_SEARCH_RANGE_FOR_TOOL_PAIRS = 5


class MessageSummarizer:
    """Summarizes messages when token limits are approached.
    
    This class mirrors the SummarizationMiddleware from LangChain v1 but is designed
    for use with custom LangGraph StateGraph implementations. It monitors message token
    counts and automatically summarizes older messages when a threshold is reached,
    preserving recent messages and maintaining context continuity by ensuring AI/Tool
    message pairs remain together.
    
    Key features (aligned with SummarizationMiddleware):
    - Configurable token counter function
    - Safe cutoff point detection to preserve AI/Tool message pairs
    - Configurable summary prompt and prefix
    - Both sync and async summarization support
    """

    def __init__(
        self,
        model: BaseChatModel,
        max_tokens_before_summary: int,
        messages_to_keep: int = _DEFAULT_MESSAGES_TO_KEEP,
        token_counter: TokenCounter = count_tokens_approximately,
        summary_prompt: str = context_summarization_prompt,
    ) -> None:
        """Initialize message summarizer.

        Args:
            model: The language model to use for generating summaries.
            max_tokens_before_summary: Token threshold to trigger summarization.
            messages_to_keep: Number of recent messages to preserve after summarization.
            token_counter: Function to count tokens in messages. Defaults to count_tokens_approximately.
            summary_prompt: Prompt template for generating summaries.
        """
        self.model = model
        self.max_tokens_before_summary = max_tokens_before_summary
        self.messages_to_keep = messages_to_keep
        self.token_counter = token_counter
        self.summary_prompt = summary_prompt

    async def before_model(self, messages: List[AnyMessage]) -> Dict[str, Any] | None:
        """Process messages before model invocation, potentially triggering summarization.
        
        Args:
            messages: List of messages to potentially summarize
            
        Returns:
            State updates with RemoveMessage pattern if summarization triggered, None otherwise
        """
        if not messages:
            return None
            
        self._ensure_message_ids(messages)
        total_tokens = self.token_counter(messages)
        
        if total_tokens < self.max_tokens_before_summary:
            return None  # No changes needed
            
        cutoff_index = self._find_safe_cutoff(messages)
        
        if cutoff_index <= 0:
            return None  # Can't summarize
            
        messages_to_summarize, preserved_messages = self._partition_messages(messages, cutoff_index)
        
        summary = await self._create_summary_async(messages_to_summarize)
        new_messages = self._build_new_messages(summary)
        
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages,
                *preserved_messages,
            ]
        }

    def _build_new_messages(self, summary: str) -> List[HumanMessage]:
        """Build new messages with summary."""
        return [
            HumanMessage(content=f"Here is a summary of the conversation to date:\n\n{summary}")
        ]

    def _ensure_message_ids(self, messages: List[AnyMessage]) -> None:
        """Ensure all messages have unique IDs."""
        for msg in messages:
            if msg.id is None:
                msg.id = str(uuid.uuid4())

    def _partition_messages(
        self,
        conversation_messages: List[AnyMessage],
        cutoff_index: int,
    ) -> tuple[List[AnyMessage], List[AnyMessage]]:
        """Partition messages into those to summarize and those to preserve."""
        messages_to_summarize = conversation_messages[:cutoff_index]
        preserved_messages = conversation_messages[cutoff_index:]
        return messages_to_summarize, preserved_messages

    def _find_safe_cutoff(self, messages: List[AnyMessage]) -> int:
        """Find safe cutoff point that preserves AI/Tool message pairs.

        Returns the index where messages can be safely cut without separating
        related AI and Tool messages. Returns 0 if no safe cutoff is found.
        """
        if len(messages) <= self.messages_to_keep:
            return 0

        target_cutoff = len(messages) - self.messages_to_keep

        for i in range(target_cutoff, -1, -1):
            if self._is_safe_cutoff_point(messages, i):
                return i

        return 0

    def _is_safe_cutoff_point(self, messages: List[AnyMessage], cutoff_index: int) -> bool:
        """Check if cutting at index would separate AI/Tool message pairs."""
        if cutoff_index >= len(messages):
            return True

        search_start = max(0, cutoff_index - _SEARCH_RANGE_FOR_TOOL_PAIRS)
        search_end = min(len(messages), cutoff_index + _SEARCH_RANGE_FOR_TOOL_PAIRS)

        for i in range(search_start, search_end):
            if not self._has_tool_calls(messages[i]):
                continue

            tool_call_ids = self._extract_tool_call_ids(cast(AIMessage, messages[i]))
            if self._cutoff_separates_tool_pair(messages, i, cutoff_index, tool_call_ids):
                return False

        return True

    def _has_tool_calls(self, message: AnyMessage) -> bool:
        """Check if message is an AI message with tool calls."""
        return bool(
            isinstance(message, AIMessage) 
            and hasattr(message, "tool_calls") 
            and message.tool_calls
        )

    def _extract_tool_call_ids(self, ai_message: AIMessage) -> set[str]:
        """Extract tool call IDs from an AI message."""
        tool_call_ids = set()
        for tc in ai_message.tool_calls:
            call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if call_id is not None:
                tool_call_ids.add(call_id)
        return tool_call_ids

    def _cutoff_separates_tool_pair(
        self,
        messages: List[AnyMessage],
        ai_message_index: int,
        cutoff_index: int,
        tool_call_ids: set[str],
    ) -> bool:
        """Check if cutoff separates an AI message from its corresponding tool messages."""
        for j in range(ai_message_index + 1, len(messages)):
            message = messages[j]
            if isinstance(message, ToolMessage) and message.tool_call_id in tool_call_ids:
                ai_before_cutoff = ai_message_index < cutoff_index
                tool_before_cutoff = j < cutoff_index
                if ai_before_cutoff != tool_before_cutoff:
                    return True
        return False

    async def _create_summary_async(self, messages_to_summarize: List[AnyMessage]) -> str:
        """Generate summary for the given messages (async version)."""
        if not messages_to_summarize:
            return "No previous conversation history."

        trimmed_messages = self._trim_messages_for_summary(messages_to_summarize)
        if not trimmed_messages:
            return "Previous conversation was too long to summarize."

        # Format messages for the summary prompt
        formatted_messages = self._format_messages_for_summary(trimmed_messages)
        
        try:
            response = await self.model.ainvoke([
                HumanMessage(content=self.summary_prompt.format(messages=formatted_messages))
            ])
            return cast(str, response.content).strip()
        except Exception as e:
            logging.error(f"Error generating summary: {e}")
            return f"Error generating summary: {str(e)}"

    def _trim_messages_for_summary(self, messages: List[AnyMessage]) -> List[AnyMessage]:
        """Trim messages to fit within summary generation limits."""
        try:
            return trim_messages(
                messages,
                max_tokens=_DEFAULT_TRIM_TOKEN_LIMIT,
                token_counter=self.token_counter,  # Use configurable token counter
                start_on="human",
                strategy="last",
                allow_partial=True,
                include_system=True,
            )
        except Exception:
            return messages[-_DEFAULT_FALLBACK_MESSAGE_COUNT:]

    def _format_messages_for_summary(self, messages: List[AnyMessage]) -> str:
        """Format messages into a string for the summary prompt."""
        formatted_parts = []
        for msg in messages:
            role = msg.__class__.__name__.replace("Message", "")
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            formatted_parts.append(f"[{role}]: {content}")
        return "\n\n".join(formatted_parts)


async def invoke_model_with_summarization(
    model: BaseChatModel,
    messages: List[AnyMessage],
    config: RunnableConfig,
    message_key: str = "messages"
) -> AIMessage:
    """Invoke model with automatic context summarization if enabled.
    
    This function wraps the model invocation pattern with automatic summarization:
    1. Checks if context summarization is enabled
    2. If enabled, summarizes messages using a dedicated summarization model
    3. Invokes the node's model with the (potentially summarized) messages
    
    Args:
        model: The model to invoke (node's task model, not summarization model)
        messages: Messages to process and pass to the model
        config: Runtime configuration
        message_key: State key for the messages (e.g., "supervisor_messages")
        
    Returns:
        AIMessage response from the model
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Apply context summarization if enabled
    if configurable.enable_context_summarization and messages:
        summarizer_model_config = build_model_config(
            configurable.context_summarization_model,
            8192,
            config,
            ["langsmith:nostream"]
        )
        summarizer = MessageSummarizer(
            model=init_chat_model(**summarizer_model_config),
            max_tokens_before_summary=configurable.max_tokens_before_summary,
            messages_to_keep=configurable.messages_to_keep,
        )
        
        # Get state updates from before_model
        state_updates = await summarizer.before_model(messages)
        
        # If summarization occurred, extract the updated messages
        if state_updates and "messages" in state_updates:
            # Filter out RemoveMessage and get the actual messages
            updated_messages = [
                msg for msg in state_updates["messages"]
                if not isinstance(msg, RemoveMessage)
            ]
            messages = updated_messages
    
    # Invoke the node's model with potentially summarized messages
    response = await model.ainvoke(messages)
    return response


##########################
# Tavily Search Tool Utils
##########################
TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events."
)
@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None
) -> str:
    """
    Fetches results from Tavily search API.

    Args
        queries (List[str]): List of search queries, you can pass in as many queries as you need.
        max_results (int): Maximum number of results to return
        topic (Literal['general', 'news', 'finance']): Topic to filter results by

    Returns:
        str: A formatted string of search results
    """
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        config=config
    )
    # Format the search results and deduplicate results by URL
    formatted_output = f"Search results: \n\n"
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}
    configurable = Configuration.from_runnable_config(config)
    max_char_to_include = 50_000   # NOTE: This can be tuned by the developer. This character count keeps us safely under input token limits for the latest models.
    model_config = build_model_config(
        configurable.summarization_model,
        configurable.summarization_model_max_tokens,
        config,
        ["langsmith:nostream"]
    )
    summarization_model = init_chat_model(**model_config).with_structured_output(Summary).with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    async def noop():
        return None
    summarization_tasks = [
        noop() if not result.get("raw_content") else summarize_webpage(
            summarization_model, 
            result['raw_content'][:max_char_to_include],
            url,
        )
        for url, result in unique_results.items()
    ]
    summaries = await asyncio.gather(*summarization_tasks)
    summarized_results = {
        url: {'title': result['title'], 'content': result['content'] if summary is None else summary}
        for url, result, summary in zip(unique_results.keys(), unique_results.values(), summaries)
    }
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"
    if summarized_results:
        return formatted_output
    else:
        return "No valid search results found. Please try different search queries or use a different search API."


async def tavily_search_async(search_queries, max_results: int = 5, topic: Literal["general", "news", "finance"] = "general", include_raw_content: bool = True, config: RunnableConfig = None):
    tavily_async_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))
    search_tasks = []
    for query in search_queries:
            search_tasks.append(
                tavily_async_client.search(
                    query,
                    max_results=max_results,
                    include_raw_content=include_raw_content,
                    topic=topic
                )
            )
    search_docs = await asyncio.gather(*search_tasks)
    return search_docs

async def summarize_webpage(model: BaseChatModel, webpage_content: str, url: str) -> str:
    try:
        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=summarize_webpage_prompt.format(webpage_content=webpage_content, url=url, date=get_today_str()))]),
            timeout=60.0
        )
        return f"""<url>\n{summary.url}\n</url>\n\n<summary>\n{summary.summary}\n</summary>\n\n<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"""
    except (asyncio.TimeoutError, Exception) as e:
        print(f"Failed to summarize webpage: {str(e)}")
        return webpage_content


##########################
# MCP Utils
##########################
async def get_mcp_access_token(
    supabase_token: str,
    base_mcp_url: str,
) -> Optional[Dict[str, Any]]:
    try:
        form_data = {
            "client_id": "mcp_default",
            "subject_token": supabase_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                base_mcp_url.rstrip("/") + "/oauth/token",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data=form_data,
            ) as token_response:
                if token_response.status == 200:
                    token_data = await token_response.json()
                    return token_data
                else:
                    response_text = await token_response.text()
                    logging.error(f"Token exchange failed: {response_text}")
    except Exception as e:
        logging.error(f"Error during token exchange: {e}")
    return None

async def get_tokens(config: RunnableConfig):
    store = get_store()
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return None
    tokens = await store.aget((user_id, "tokens"), "data")
    if not tokens:
        return None
    expires_in = tokens.value.get("expires_in")  # seconds until expiration
    created_at = tokens.created_at  # datetime of token creation
    current_time = datetime.now(timezone.utc)
    expiration_time = created_at + timedelta(seconds=expires_in)
    if current_time > expiration_time:
        await store.adelete((user_id, "tokens"), "data")
        return None

    return tokens.value

async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    store = get_store()
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return
    await store.aput((user_id, "tokens"), "data", tokens)
    return

async def fetch_tokens(config: RunnableConfig) -> dict[str, Any]:
    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens
    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if not supabase_token:
        return None
    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not mcp_config or not mcp_config.get("url"):
        return None
    mcp_tokens = await get_mcp_access_token(supabase_token, mcp_config.get("url"))

    await set_tokens(config, mcp_tokens)
    return mcp_tokens

def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    old_coroutine = tool.coroutine
    async def wrapped_mcp_coroutine(**kwargs):
        def _find_first_mcp_error_nested(exc: BaseException) -> McpError | None:
            if isinstance(exc, McpError):
                return exc
            if isinstance(exc, ExceptionGroup):
                for sub_exc in exc.exceptions:
                    if found := _find_first_mcp_error_nested(sub_exc):
                        return found
            return None
        try:
            return await old_coroutine(**kwargs)
        except BaseException as e_orig:
            mcp_error = _find_first_mcp_error_nested(e_orig)
            if not mcp_error:
                raise e_orig
            error_details = mcp_error.error
            is_interaction_required = getattr(error_details, "code", None) == -32003
            error_data = getattr(error_details, "data", None) or {}
            if is_interaction_required:
                message_payload = error_data.get("message", {})
                error_message_text = "Required interaction"
                if isinstance(message_payload, dict):
                    error_message_text = (
                        message_payload.get("text") or error_message_text
                    )
                if url := error_data.get("url"):
                    error_message_text = f"{error_message_text} {url}"
                raise ToolException(error_message_text) from e_orig
            raise e_orig
    tool.coroutine = wrapped_mcp_coroutine
    return tool

async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    configurable = Configuration.from_runnable_config(config)
    if configurable.mcp_config and configurable.mcp_config.auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None
    if not (configurable.mcp_config and configurable.mcp_config.url and configurable.mcp_config.tools and (mcp_tokens or not configurable.mcp_config.auth_required)):
        return []
    tools = []
    # When the Multi-MCP Server support is merged in OAP, update this code.
    server_url = configurable.mcp_config.url.rstrip("/") + "/mcp"
    mcp_server_config = {
        "server_1":{
            "url": server_url,
            "headers": {"Authorization": f"Bearer {mcp_tokens['access_token']}"} if mcp_tokens else None,
            "transport": "streamable_http"
        }
    }
    try:
        client = MultiServerMCPClient(mcp_server_config)
        mcp_tools = await client.get_tools()
    except Exception as e:
        print(f"Error loading MCP tools: {e}")
        return []
    for tool in mcp_tools:
        if tool.name in existing_tool_names:
            warnings.warn(
                f"Trying to add MCP tool with a name {tool.name} that is already in use - this tool will be ignored."
            )
            continue
        if tool.name not in set(configurable.mcp_config.tools):
            continue
        tools.append(wrap_mcp_authenticate_tool(tool))
    return tools


##########################
# Tool Utils
##########################
async def get_search_tool(search_api: SearchAPI):
    if search_api == SearchAPI.ANTHROPIC:
        return [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
    elif search_api == SearchAPI.OPENAI:
        return [{"type": "web_search_preview"}]
    elif search_api == SearchAPI.TAVILY:
        search_tool = tavily_search
        search_tool.metadata = {**(search_tool.metadata or {}), "type": "search", "name": "web_search"}
        return [search_tool]
    elif search_api == SearchAPI.NONE:
        return []
    
async def get_all_tools(config: RunnableConfig):
    tools = [tool(ResearchComplete)]
    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    tools.extend(await get_search_tool(search_api))
    existing_tool_names = {tool.name if hasattr(tool, "name") else tool.get("name", "web_search") for tool in tools}
    mcp_tools = await load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    return tools

def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]


##########################
# Model Provider Native Websearch Utils
##########################
def anthropic_websearch_called(response):
    try:
        usage = response.response_metadata.get("usage")
        if not usage:
            return False
        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False
        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False
        return web_search_requests > 0
    except (AttributeError, TypeError):
        return False

def openai_websearch_called(response):
    tool_outputs = response.additional_kwargs.get("tool_outputs")
    if tool_outputs:
        for tool_output in tool_outputs:
            if tool_output.get("type") == "web_search_call":
                return True
    return False


##########################
# Token Limit Exceeded Utils
##########################
def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    error_str = str(exception).lower()
    provider = None
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith('openai:'):
            provider = 'openai'
        elif model_str.startswith('anthropic:'):
            provider = 'anthropic'
        elif model_str.startswith('gemini:') or model_str.startswith('google:'):
            provider = 'gemini'
    if provider == 'openai':
        return _check_openai_token_limit(exception, error_str)
    elif provider == 'anthropic':
        return _check_anthropic_token_limit(exception, error_str)
    elif provider == 'gemini':
        return _check_gemini_token_limit(exception, error_str)
    
    return (_check_openai_token_limit(exception, error_str) or
            _check_anthropic_token_limit(exception, error_str) or
            _check_gemini_token_limit(exception, error_str))

def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    is_openai_exception = ('openai' in exception_type.lower() or 
                          'openai' in module_name.lower())
    is_bad_request = class_name in ['BadRequestError', 'InvalidRequestError']
    if is_openai_exception and is_bad_request:
        token_keywords = ['token', 'context', 'length', 'maximum context', 'reduce']
        if any(keyword in error_str for keyword in token_keywords):
            return True
    if hasattr(exception, 'code') and hasattr(exception, 'type'):
        if (getattr(exception, 'code', '') == 'context_length_exceeded' or
            getattr(exception, 'type', '') == 'invalid_request_error'):
            return True
    return False

def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    is_anthropic_exception = ('anthropic' in exception_type.lower() or 
                             'anthropic' in module_name.lower())
    is_bad_request = class_name == 'BadRequestError'
    if is_anthropic_exception and is_bad_request:
        if 'prompt is too long' in error_str:
            return True
    return False

def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    is_google_exception = ('google' in exception_type.lower() or 'google' in module_name.lower())
    is_resource_exhausted = class_name in ['ResourceExhausted', 'GoogleGenerativeAIFetchError']
    if is_google_exception and is_resource_exhausted:
        return True
    if 'google.api_core.exceptions.resourceexhausted' in exception_type.lower():
        return True
    
    return False

# NOTE: This may be out of date or not applicable to your models. Please update this as needed.
MODEL_TOKEN_LIMITS = {
    "openai:gpt-4.1-mini": 1047576,
    "openai:gpt-4.1-nano": 1047576,
    "openai:gpt-4.1": 1047576,
    "openai:gpt-4o-mini": 128000,
    "openai:gpt-4o": 128000,
    "openai:o4-mini": 200000,
    "openai:o3-mini": 200000,
    "openai:o3": 200000,
    "openai:o3-pro": 200000,
    "openai:o1": 200000,
    "openai:o1-pro": 200000,
    "azure_openai:o4-mini": 200000,
    "azure_openai:o3": 200000,
    "azure_openai:gpt-5": 200000,
    "anthropic:claude-opus-4": 200000,
    "anthropic:claude-sonnet-4": 200000,
    "anthropic:claude-3-7-sonnet": 200000,
    "anthropic:claude-3-5-sonnet": 200000,
    "anthropic:claude-3-5-haiku": 200000,
    "google:gemini-1.5-pro": 2097152,
    "google:gemini-1.5-flash": 1048576,
    "google:gemini-pro": 32768,
    "cohere:command-r-plus": 128000,
    "cohere:command-r": 128000,
    "cohere:command-light": 4096,
    "cohere:command": 4096,
    "mistral:mistral-large": 32768,
    "mistral:mistral-medium": 32768,
    "mistral:mistral-small": 32768,
    "mistral:mistral-7b-instruct": 32768,
    "ollama:codellama": 16384,
    "ollama:llama2:70b": 4096,
    "ollama:llama2:13b": 4096,
    "ollama:llama2": 4096,
    "ollama:mistral": 32768,
}

def get_model_token_limit(model_string):
    for key, token_limit in MODEL_TOKEN_LIMITS.items():
        if key in model_string:
            return token_limit
    return None

def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            return messages[:i]  # Return everything up to (but not including) the last AI message
    return messages

##########################
# Misc Utils
##########################
def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")

def get_config_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value

def get_api_key_for_model(model_name: str, config: RunnableConfig):
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    model_name = model_name.lower()
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        if model_name.startswith("openai:"):
            return api_keys.get("OPENAI_API_KEY")
        elif model_name.startswith("azure_openai:"):
            return api_keys.get("AZURE_OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return api_keys.get("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return api_keys.get("GOOGLE_API_KEY")
        return None
    else:
        if model_name.startswith("openai:"): 
            return os.getenv("OPENAI_API_KEY")
        elif model_name.startswith("azure_openai:"):
            return os.getenv("AZURE_OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return os.getenv("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return os.getenv("GOOGLE_API_KEY")
        return None

def get_tavily_api_key(config: RunnableConfig):
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        return api_keys.get("TAVILY_API_KEY")
    else:
        return os.getenv("TAVILY_API_KEY")

def build_model_config(model_name: str, max_tokens: int, config: RunnableConfig, tags: List[str] = None):
    """Build model configuration with Azure OpenAI parameters when needed."""
    from .configuration import Configuration
    
    configurable = Configuration.from_runnable_config(config)
    base_config = {
        "api_key": get_api_key_for_model(model_name, config),
    }
    
    if tags:
        base_config["tags"] = tags
    
    # Add Azure-specific parameters if this is an Azure OpenAI model
    if model_name.lower().startswith("azure_openai:"):
        # Extract the actual model name (remove azure_openai: prefix)
        actual_model = model_name.lower().split(":")[-1]
        
        base_config["model"] = actual_model  # Use actual model name (e.g., "o3", "o4-mini")
        base_config["model_provider"] = "azure_openai"
        base_config["azure_endpoint"] = configurable.azure_openai_endpoint
        base_config["api_version"] = configurable.azure_openai_api_version
        
        # Map model name to deployment
        if "o4-mini" in actual_model:
            base_config["azure_deployment"] = configurable.azure_o4_mini_deployment
        elif "o3" in actual_model:
            base_config["azure_deployment"] = configurable.azure_o3_deployment
        elif "gpt-5" in actual_model:
            base_config["azure_deployment"] = configurable.azure_gpt5_deployment
            base_config["api_version"] = configurable.gpt5_api_version
    else:
        base_config["model"] = model_name
    
    return base_config

@tool
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.
    
    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.
    
    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?
    
    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?
    
    Args:
        reflection: Detailed reflection on research progress, findings, gaps, and next steps
        
    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"

@tool
def refine_draft_report(
    research_brief: Annotated[str, InjectedToolArg],
    findings: Annotated[str, InjectedToolArg], 
    draft_report: Annotated[str, InjectedToolArg],
    config: RunnableConfig = None
) -> str:
    """Refine draft report using research findings.
    
    Synthesizes all research findings into a comprehensive draft report.
    
    Args:
        research_brief: User's research request
        findings: Collected research findings for the user request
        draft_report: Draft report based on the findings and user request
        config: Runtime configuration
        
    Returns:
        Refined draft report
    """
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = build_model_config(
        configurable.final_report_model,
        configurable.final_report_model_max_tokens,
        config
    )
    writer_model = init_chat_model(**writer_model_config)
    
    draft_report_prompt = report_generation_with_draft_insight_prompt.format(
        research_brief=research_brief,
        findings=findings,
        draft_report=draft_report,
        date=get_today_str()
    )
    
    response = writer_model.invoke([HumanMessage(content=draft_report_prompt)])
    
    return response.content