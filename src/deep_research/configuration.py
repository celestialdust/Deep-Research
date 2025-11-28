from pydantic import BaseModel, Field
from typing import Any, List, Optional
from langchain_core.runnables import RunnableConfig
import os
from enum import Enum

class SearchAPI(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"

class MCPConfig(BaseModel):
    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """The URL of the MCP server"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """The tools to make available to the LLM"""
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """Whether the MCP server requires authentication"""

class Configuration(BaseModel):
    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of retries for structured output calls from models"
            }
        }
    )
    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to allow the researcher to ask the user clarifying questions before starting research"
            }
        }
    )
    max_concurrent_research_units: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits."
            }
        }
    )
    enable_human_in_the_loop: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Whether to enable human-in-the-loop approval for the research brief before proceeding to draft report generation"
            }
        }
    )
    max_brief_refinement_rounds: int = Field(
        default=1,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 1,
                "min": 1,
                "max": 10,
                "description": "Maximum number of times the research brief can be refined based on human feedback"
            }
        }
    )
    # Research Configuration
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "tavily",
                "description": "Search API to use for research. NOTE: Make sure your Researcher Model supports the selected search API.",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "OpenAI Native Web Search", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic Native Web Search", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "None", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 3,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions."
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "Maximum number of tool calling iterations to make in a single researcher step."
            }
        }
    )
    enable_search_for_brief: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Whether to enable Tavily search for the research brief generation node (max 1 search allowed)"
            }
        }
    )
    enable_search_for_draft: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Whether to enable Tavily search for the draft report generation node (max 1 search allowed)"
            }
        }
    )
    
    # Azure OpenAI Configuration
    azure_openai_endpoint: str = Field(
        default="https://DM-OPENAI-DEV-SWEDEN.openai.azure.com",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "https://DM-OPENAI-DEV-SWEDEN.openai.azure.com",
                "description": "Azure OpenAI endpoint URL"
            }
        }
    )
    azure_openai_api_version: str = Field(
        default="2025-01-01-preview",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "2025-01-01-preview",
                "description": "Azure OpenAI API version"
            }
        }
    )
    azure_o4_mini_deployment: str = Field(
        default="gsds-o4-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "gsds-o4-mini",
                "description": "Azure deployment name for o4-mini model"
            }
        }
    )
    azure_o3_deployment: str = Field(
        default="gsds-o3",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "gsds-o3",
                "description": "Azure deployment name for o3 model"
            }
        }
    )
    gpt5_api_version: str = Field(
        default="2025-04-01-preview",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "2025-04-01-preview",
                "description": "Azure OpenAI API version for GPT-5 model"
            }
        }
    )
    azure_gpt5_deployment: str = Field(
        default="gsds-gpt-5",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "gsds-gpt-5",
                "description": "Azure deployment name for GPT-5 model"
            }
        }
    )
    
    # Model Configuration
    summarization_model: str = Field(
        default="azure_openai:gpt-5",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "azure_openai:o3",
                "description": "Model for summarizing research results from OpenAI search results"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for summarization model"
            }
        }
    )
    research_model: str = Field(
        default="azure_openai:gpt-5",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "azure_openai:o3",
                "description": "Model for conducting research. NOTE: Make sure your Researcher Model supports the selected search API."
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for research model"
            }
        }
    )
    compression_model: str = Field(
        default="azure_openai:o4-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "azure_openai:o4-mini",
                "description": "Model for compressing research findings from sub-agents. NOTE: Make sure your Compression Model supports the selected search API."
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for compression model"
            }
        }
    )
    final_report_model: str = Field(
        default="azure_openai:gpt-5",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "azure_openai:o3",
                "description": "Model for writing the final report from all research findings"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for final report model"
            }
        }
    )

    # MCP server configuration for the deep research agent
    # see load_mcp_tools in utils.py for more details
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP server configuration"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Any additional instructions to pass along to the Agent regarding the MCP tools that are available to it."
            }
        }
    )

    # Context Summarization Configuration
    enable_context_summarization: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to enable automatic context summarization when approaching token limits"
            }
        }
    )
    max_tokens_before_summary: int = Field(
        default=100000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 100000,
                "min": 10000,
                "max": 200000,
                "description": "Token threshold to trigger context summarization. When messages exceed this limit, older messages will be summarized."
            }
        }
    )
    messages_to_keep: int = Field(
        default=20,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 20,
                "min": 5,
                "max": 50,
                "description": "Number of recent messages to preserve after summarization"
            }
        }
    )
    context_summarization_model: str = Field(
        default="azure_openai:o4-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "azure_openai:o4-mini",
                "description": "Model to use for context summarization (o4-mini recommended for speed)"
            }
        }
    )


    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        arbitrary_types_allowed = True