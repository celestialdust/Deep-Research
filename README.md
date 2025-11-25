# Deep Research

### Quickstart

1. Clone the repository and activate a virtual environment:
```bash
git clone <the repo url>
cd open_deep_research
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
uv pip install -r pyproject.toml
```

3. Set up your `.env` file to customize the environment variables (for model selection, search tools, and other configuration settings):
```bash
cp .env.example .env
```

4. Launch the assistant with the LangGraph server locally to open LangGraph Studio in your browser:

```bash
# Install dependencies and start the LangGraph server
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

Use this to open the Studio UI:
```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```
Ask a question in the `messages` input field and click `Submit`.

### Configurations

Deep Research offers extensive configuration options to customize the research process and model behavior. All configurations can be set via the web UI, environment variables, or by modifying the configuration directly.

#### General Settings

- **Max Structured Output Retries** (default: 3): Maximum number of retries for structured output calls from models when parsing fails
- **Allow Clarification** (default: true): Whether to allow the researcher to ask clarifying questions before starting research
- **Max Concurrent Research Units** (default: 5): Maximum number of research units to run concurrently using sub-agents. Higher values enable faster research but may hit rate limits

#### Research Configuration

- **Search API** (default: Tavily): Choose from Tavily (works with all models), OpenAI Native Web Search, Anthropic Native Web Search, or None
- **Max Researcher Iterations** (default: 3): Number of times the Research Supervisor will reflect on research and ask follow-up questions
- **Max React Tool Calls** (default: 5): Maximum number of tool calling iterations in a single researcher step

## Architecture Overview

Deep Research Agent uses a sophisticated multi-agent architecture built on LangGraph with the following components:

- **Research Supervisor**: Orchestrates the research process using a diffusion algorithm, breaks down complex queries, and manages parallel research units with strategic reflection via `think_tool`
- **Draft Report Generator**: Creates an initial draft report from the research brief that serves as a baseline for iterative refinement
- **Research Agents**: Conduct focused research on specific topics using available tools and search APIs
- **Compression Agent**: Synthesizes and compresses research findings from multiple agents
- **Draft Report Refiner**: Iteratively refines the draft report based on new research findings using the diffusion algorithm
- **Report Generator**: Creates comprehensive final reports from all research findings with enhanced insightfulness and helpfulness criteria

### Key Features

- **Diffusion Algorithm**: Iterative denoising approach where research progressively refines an initial draft by identifying gaps, conducting targeted research, and updating the report
- **Strategic Reflection**: `think_tool` enables the supervisor to pause and reflect on research progress before making decisions
- **Multi-Language Support**: Automatically detects and responds in the same language as the user's input
- **Enhanced Quality Criteria**: Final reports follow strict insightfulness (granular breakdowns, detailed tables, nuanced discussion) and helpfulness (satisfying intent, clarity, accuracy) rules

## Project Structure

```
deep_research_openai/
├── src/
│   ├── deep_research/              # Core research agent package
│   │   ├── __init__.py            # Package initialization
│   │   ├── configuration.py      # Configuration classes and settings
│   │   ├── deep_researcher.py     # Main agent implementation and LangGraph workflow
│   │   ├── prompts.py            # System prompts and templates
│   │   ├── state.py              # State management and data models
│   │   └── utils.py              # Utility functions, tool loading, and MCP integration
│   └── security/
│       └── auth.py               # Supabase authentication for LangGraph Studio
├── tests/                        # Evaluation framework (in development)
│   ├── evaluators.py            # Quality assessment evaluators
│   ├── pairwise_evaluation.py   # Head-to-head model comparisons
│   ├── prompts.py              # Evaluation prompts and criteria
│   ├── run_evaluate.py         # Main evaluation runner
│   └── supervisor_parallel_evaluation.py  # Parallelism testing
├── langgraph.json              # LangGraph configuration
├── pyproject.toml             # Project dependencies and metadata
└── test-agent.ipynb          # Interactive testing notebook
```

### Core Components Explained

#### `src/deep_research/`

- **`configuration.py`**: Defines the `Configuration` class with all configurable parameters including model settings, search APIs, MCP configuration, and Azure OpenAI settings. Supports both environment variables and UI configuration.

- **`deep_researcher.py`**: Contains the main LangGraph workflow implementation with nodes for:
  - User clarification
  - Research brief generation
  - Draft report generation
  - Research supervisor (with diffusion algorithm)
  - Parallel research execution
  - Research compression
  - Draft report refinement
  - Final report generation

- **`state.py`**: Defines all state classes and data models:
  - `AgentState`: Main workflow state (includes `draft_report` field)
  - `SupervisorState`: Research supervisor state (includes `draft_report` field)
  - `ResearcherState`: Individual researcher state
  - `DraftReport`: Structured output for draft report generation
  - Structured outputs for research coordination

- **`utils.py`**: Utility functions including:
  - `think_tool`: Strategic reflection tool for research planning
  - `refine_draft_report`: Tool for iteratively refining the draft report
  - MCP tool loading and authentication
  - Search API integration (Tavily, OpenAI, Anthropic)
  - Model configuration builders
  - Token limit management

- **`prompts.py`**: System prompts and templates for:
  - Research clarification
  - Research planning with diffusion algorithm
  - Draft report generation
  - Research execution with strategic reflection
  - Content summarization and compression (with research topic context)
  - Draft report refinement
  - Final report generation (with insightfulness/helpfulness criteria and multi-language support)

#### `src/security/`

- **`auth.py`**: Supabase-based authentication system for LangGraph Studio with user isolation and access control.

## MCP Integration

Deep Research Agent supports custom MCP (Model Context Protocol) servers, allowing you to extend the research capabilities with custom tools.

### MCP Configuration

Configure MCP integration via the web UI or environment variables:

```json
{
  "url": "https://your-mcp-server.com",
  "tools": ["custom_search", "data_analyzer", "document_processor"],
  "auth_required": true
}
```

### MCP Configuration Parameters

- **`url`**: Your MCP server endpoint URL
- **`tools`**: List of specific tools to make available to the research agent
- **`auth_required`**: Whether the MCP server requires authentication

### Setting Up Custom MCP Tools

1. **Deploy your MCP server** with the tools you want to integrate
2. **Configure authentication** (if required) - the system supports Bearer token authentication
3. **Set the MCP configuration** in the web UI or environment variables
4. **Add tool instructions** via the `mcp_prompt` field to guide the agent on how to use your custom tools

### Example MCP Integration

```python
# Example MCP configuration for a custom research tool
mcp_config = {
    "url": "https://my-research-tools.com/mcp",
    "tools": ["scientific_paper_search", "patent_analyzer", "market_data_fetcher"],
    "auth_required": True
}

# Optional: Provide instructions for the agent
mcp_prompt = """
You have access to specialized research tools:
- scientific_paper_search: Use for academic research queries
- patent_analyzer: Use for patent and IP research
- market_data_fetcher: Use for market analysis and financial data
"""
```

The agent will automatically load and integrate your custom MCP tools alongside the built-in search capabilities, providing a unified research experience.

## Future Developments

### Engineering

- Try different search apis and see which one works best
- Reinforcement learning for better dynamic reasoning/planning + tool use
- Web Search Resource validation agent
- Apply Context Engineering practices
- Integrate domain specific insights from feedbacks

### Product/UX

- Create a separate UI for deep research while putting it as a add-on in clinical trial protocol copilot
- Send deep research reports as attachments to users in email after deep research is done


