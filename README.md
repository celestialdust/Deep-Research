# Deep Research

### Quickstart

1. Clone the repository and activate a virtual environment:
```bash
git clone <the repo url>
cd deep_research_openai
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
- **Allow Clarification** (default: false): Whether to allow the researcher to ask clarifying questions before starting research
- **Max Concurrent Research Units** (default: 5): Maximum number of research units to run concurrently using sub-agents. Higher values enable faster research but may hit rate limits
- **Enable Human-in-the-Loop** (default: true): Whether to enable human approval for the research brief before proceeding to draft report generation
- **Max Brief Refinement Rounds** (default: 1): Maximum number of times the research brief can be refined based on human feedback before proceeding automatically

#### Research Configuration

- **Search API** (default: Tavily): Choose from Tavily (works with all models), OpenAI Native Web Search, Anthropic Native Web Search, or None
- **Max Researcher Iterations** (default: 6): Number of times the Research Supervisor will reflect on research and ask follow-up questions
- **Max React Tool Calls** (default: 4): Maximum number of tool calling iterations in a single researcher step
- **Enable Search for Brief** (default: false): Whether to enable Tavily search for the research brief generation node (max 1 search allowed)
- **Enable Search for Draft** (default: true): Whether to enable Tavily search for the draft report generation node (max 1 search allowed)

#### Model Configuration

- **Supported Models**: GPT-5, O3, O4-mini and other Azure OpenAI models
- **GPT-5 Configuration**: 
  - Deployment name: `gsds-gpt-5` (default)
  - API Version: `2025-04-01-preview`
  - Token limit: 200,000 tokens
- **Model Selection**: Configure different models for summarization, research, compression, final report generation, citation checking, and PDF conversion
  - `summarization_model` (default: azure_openai:o4-mini): For summarizing search results
  - `research_model` (default: azure_openai:gpt-5): For conducting research
  - `compression_model` (default: azure_openai:o4-mini): For compressing research findings
  - `final_report_model` (default: azure_openai:gpt-5): For generating final reports
- **Azure OpenAI Settings**: Customize endpoint, API version, and deployment names for each model

#### Citation & PDF Configuration

- **Citation Check Model** (default: azure_openai:gpt-5): Model for verifying citations match source text exactly
- **Citation Check Model Max Tokens** (default: 16,000): Maximum output tokens for citation check model
- **Citation Check Tool Call Limit** (default: 20): Maximum number of tool calls for citation check agent per run
- **PDF Conversion Model** (default: azure_openai:gpt-5): Model for converting `<ref>` tags to text-fragment URLs
- **PDF Conversion Model Max Tokens** (default: 16,000): Maximum output tokens for PDF conversion model

#### Context Engineering Configuration

- **Enable Context Summarization** (default: true): Whether to enable automatic context summarization when approaching token limits
- **Max Tokens Before Summary** (default: 100,000): Token threshold to trigger context summarization. When messages exceed this limit, older messages will be summarized
- **Messages to Keep** (default: 20): Number of recent messages to preserve after summarization
- **Context Summarization Model** (default: azure_openai:o4-mini): Model to use for context summarization (o4-mini recommended for speed)

## Architecture Overview

Deep Research Agent uses a sophisticated multi-agent architecture built on LangGraph with the following components:

- **User Clarification** (`clarify_with_user`): Optionally asks clarifying questions before starting research
- **Research Brief Generator** (`write_research_brief`): Creates a detailed research brief from user messages, with optional Tavily search for up-to-date context
- **Human-in-the-Loop Approval** (`approve_research_brief`): Allows human review and refinement of the research brief before proceeding (uses LangGraph interrupt pattern)
- **Draft Report Generator** (`write_draft_report`): Creates an initial draft report from the research brief that serves as a baseline for iterative refinement, with optional Tavily search for preliminary context
- **Research Supervisor** (`supervisor_subgraph`): Orchestrates the research process using a diffusion algorithm, breaks down complex queries, and manages parallel research units with strategic reflection via `think_tool`
- **Research Agents** (`researcher_subgraph`): Conduct focused research on specific topics using available tools and search APIs
- **Compression Agent** (`compress_research`): Synthesizes and compresses research findings from multiple agents while preserving exact source sentences
- **Draft Report Refiner** (`refine_draft_report` tool): Iteratively refines the draft report based on new research findings using the diffusion algorithm
- **Report Generator** (`final_report_generation`): Creates comprehensive final reports from all research findings with enhanced insightfulness and helpfulness criteria
- **Post-Processing Pipeline** (`post_process_report` wrapper → `post_processing_subgraph`):
  - **Citation Check Agent** (`pp_check_citations`): Verifies that all `<ref>` citations match research notes exactly using `create_agent` with `citation_think_tool`
  - **URL Conversion Agent** (`pp_convert_refs_to_urls`): LLM converts `<ref id="N">"source"</ref>` tags to text-fragment URLs using range matching
  - **PDF Generator** (`pp_convert_to_pdf`): Converts markdown to PDF with WeasyPrint

### Key Features

- **Diffusion Algorithm**: Iterative denoising approach where research progressively refines an initial draft by identifying gaps, conducting targeted research, and updating the report
- **Human-in-the-Loop**: Optional human approval workflow for research briefs with configurable refinement rounds
- **Contextual Search**: Optional Tavily search integration for research brief and draft report generation to ensure up-to-date information
- **Strategic Reflection**: `think_tool` enables the supervisor to pause and reflect on research progress before making decisions
- **Multi-Model Support**: Support for GPT-5, O3, O4-mini and other Azure OpenAI models with flexible configuration
- **Multi-Language Support**: Automatically detects and responds in the same language as the user's input
- **Enhanced Quality Criteria**: Final reports follow strict insightfulness (granular breakdowns, detailed tables, nuanced discussion) and helpfulness (satisfying intent, clarity, accuracy) rules
- **Context Engineering**: Automatic context window summarization using o4-mini when approaching token limits, mirroring LangChain v1's `SummarizationMiddleware`
- **Research Validation**: Notes and raw notes are asynchronously saved to `/docs` before final report generation for resource validation (non-blocking I/O)
- **Retraceable Citations**: Text-fragment links using range matching enable one-click navigation to the exact source sentence, with robust URL utilities and PDF-compatible short URLs (see [Citations](#retraceable-citations))
- **PDF Export**: Automatic conversion of final reports to professional PDF format with serif fonts and proper margins (see [PDF Export](#pdf-export))

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
├── docs/                         # Research notes output directory
│   └── .gitkeep                 # Placeholder (note files are gitignored)
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

- **`configuration.py`**: Defines the `Configuration` class with all configurable parameters including:
  - Model settings (GPT-5, O3, O4-mini support)
  - Search APIs (Tavily, OpenAI, Anthropic)
  - Human-in-the-loop configuration
  - Search enablement for brief and draft nodes
  - Context summarization settings (token threshold, messages to keep, summarization model)
  - Citation check model settings (for verifying citations against research notes)
  - PDF conversion model settings (for converting refs to text-fragment URLs)
  - MCP configuration
  - Azure OpenAI settings with separate deployment configurations
  - Supports both environment variables and UI configuration

- **`deep_researcher.py`**: Contains the main LangGraph workflow implementation with:
  - **Main Graph Nodes**:
    - `clarify_with_user`: Optional user clarification before research
    - `write_research_brief`: Research brief generation (with optional Tavily search)
    - `approve_research_brief`: Human-in-the-loop research brief approval (interrupt-based)
    - `write_draft_report`: Draft report generation (with optional Tavily search and `<ref>` citations)
    - `research_supervisor`: Invokes the supervisor subgraph (with diffusion algorithm)
    - `final_report_generation`: Final report with `<ref>` inline citations and async note-saving to `/docs`
    - `post_process_report`: Wrapper node invoking post-processing subgraph with isolated state
  - **Supervisor Subgraph** (`supervisor_subgraph`):
    - `supervisor`: Research planning with think_tool and ConductResearch
    - `supervisor_tools`: Executes research via researcher subgraph, handles refine_draft_report
  - **Researcher Subgraph** (`researcher_subgraph`):
    - `researcher`: Conducts focused research with search tools
    - `researcher_tools`: Executes tool calls
    - `compress_research`: Compresses findings with exact source sentence preservation
  - **Post-Processing Subgraph** (`post_processing_subgraph`):
    - `check_citations`: Verifies citations match research notes exactly using `create_agent` with `citation_think_tool`
    - `convert_refs_to_urls`: LLM converts `<ref>` tags to text-fragment URLs for PDF
    - `convert_to_pdf`: Markdown → HTML → PDF with WeasyPrint

- **`state.py`**: Defines all state classes and data models:
  - `AgentState`: Main workflow state (includes `draft_report`, `brief_refinement_rounds`, `pdf_path`, `md_path`, `final_report_pdf` fields)
  - `AgentInputState`: Input state for the main graph (messages only)
  - `SupervisorState`: Research supervisor state (includes `draft_report`, `research_iterations` fields)
  - `ResearcherState`: Individual researcher state (includes `tool_call_iterations`, `research_topic` fields)
  - `ResearcherOutputState`: Output state for researcher subgraph
  - `PostProcessingState`: State for post-processing subgraph with isolated messages
  - `PostProcessingInputState`: Input state for post-processing (final_report, notes)
  - `PostProcessingOutputState`: Output state for post-processing (final_report, final_report_pdf, pdf_path, md_path)
  - `ClaimSourcePair`: Atomic claim with exact source sentence for text-fragment citations
  - `Summary`: Structured webpage summary with claim-source pairs
  - `ClarifyWithUser`, `ResearchQuestion`, `DraftReport`: Structured outputs for research coordination
  - `ConductResearch`, `ResearchComplete`: Tool schema classes for supervisor

- **`utils.py`**: Utility functions including:
  - `MessageSummarizer`: Context window summarization class (mirrors LangChain v1's `SummarizationMiddleware`)
  - `invoke_model_with_summarization`: Wrapper for model invocation with automatic context summarization
  - `think_tool`: Strategic reflection tool for research planning
  - `citation_think_tool`: Analysis tool for verifying citation accuracy against research notes
  - `refine_draft_report`: Tool for iteratively refining the draft report
  - `tavily_search`: Tavily search tool for research brief and draft generation
  - `summarize_webpage`: Webpage summarization with atomic claim-source pair extraction (with automatic retry on token limits)
  - PDF generation utilities: `generate_pdf_from_markdown`, `save_markdown_file`, `extract_title_from_markdown`, `sanitize_filename`, `convert_citations_to_superscript`
  - `PDF_CSS_STYLES`: Professional CSS styling for PDF reports (OpenAI-style circular citation badges)
  - MCP tool loading and authentication (`load_mcp_tools`, `fetch_tokens`, `get_mcp_access_token`)
  - Search API integration (Tavily, OpenAI, Anthropic) via `get_search_tool`, `get_all_tools`
  - Model configuration builders (`build_model_config`) with GPT-5 and Azure OpenAI support
  - Token limit management (`is_token_limit_exceeded`, `get_model_token_limit`, `MODEL_TOKEN_LIMITS`)

- **`prompts.py`**: System prompts and templates for:
  - `clarify_with_user_instructions`: Research clarification prompt
  - `transform_messages_into_research_topic_prompt`: Research brief generation (with optional search tool instructions)
  - `lead_researcher_prompt`: Research planning with diffusion algorithm (ConductResearch, refine_draft_report, ResearchComplete tools)
  - `draft_report_generation_prompt`: Draft report generation (with `<ref id="N">"source"</ref>` citation format)
  - `research_system_prompt`: Research execution with strategic reflection (think_tool, tavily_search)
  - `summarize_webpage_prompt`: Webpage summarization with atomic claim-source pair extraction (5-10 selective pairs)
  - `compress_research_system_prompt` / `compress_research_simple_human_message`: Content compression with exact source sentence preservation
  - `report_generation_with_draft_insight_prompt`: Draft report refinement
  - `final_report_generation_prompt`: Final report generation (with `<ref>` inline citations and Sources section rules)
  - `context_summarization_prompt`: Context window summarization for MessageSummarizer
  - `citation_check_prompt`: Citation verification (for checking citations against research notes)
  - `convert_refs_to_urls_prompt`: URL conversion (for converting refs to text-fragment URLs with range matching)

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

## Context Engineering

Deep Research implements context engineering practices inspired by LangChain v1's middleware system, adapted for custom LangGraph StateGraph implementations.

### MessageSummarizer

The `MessageSummarizer` class provides automatic context window management by summarizing older messages when token limits are approached. It mirrors the functionality of LangChain v1's `SummarizationMiddleware`.

**Key Features:**
- **Token-based triggering**: Summarization triggers when message tokens exceed `max_tokens_before_summary`
- **Safe cutoff detection**: Preserves AI/Tool message pairs to avoid breaking tool call chains
- **Configurable token counter**: Defaults to `count_tokens_approximately`, can be customized
- **Async support**: `before_model()` method processes messages before model invocation

**Helper Function:**
The `invoke_model_with_summarization()` function wraps model invocation with automatic context summarization, making it easy to integrate into any node.

**Applied to nodes via `invoke_model_with_summarization`:**
- `supervisor` - Summarizes `supervisor_messages`
- `researcher` - Summarizes `researcher_messages`  
- `compress_research` - Summarizes `researcher_messages`

### Research Notes Persistence

Research notes are asynchronously saved to `/docs` **after successful final report generation** to prevent duplicate files during retries:
- `docs/notes_{timestamp}.md` - Compressed research findings
- `docs/raw_notes_{timestamp}.md` - Raw tool outputs and AI responses

**Why after generation?** If notes were saved before the LLM call, LangGraph retries (e.g., due to cancellation or timeout) would create duplicate files with different timestamps.

All file operations use `asyncio.to_thread()` to prevent blocking the ASGI event loop, ensuring optimal performance in production deployments.

## Retraceable Citations

Deep Research generates citations with text-fragment links that navigate directly to the supporting sentence in source documents, enabling one-click verification of claims.

### How It Works

1. **Webpage Summarization**: When webpages are summarized, the system extracts atomic claim-source pairs for the most important key claims:
   - `claim`: A key factual claim from the summary (headline-worthy facts)
   - `source_sentence`: The exact verbatim sentence from the source
   
   **Selective Extraction**: Only 5-10 most citation-worthy claims get source sentences (headline facts, primary findings, unique insights). The summary remains comprehensive while claim-source pairs focus on the most important facts likely to be cited in final reports.
   
   The summarization includes robust error handling:
   - Automatic content truncation to stay within token limits (default 30,000 chars)
   - Retry with shorter content on token limit errors
   - Graceful fallback to raw content excerpt if summarization fails
   - Extended timeout (180 seconds initial, 120 seconds retry) for complex pages

2. **Compression Agent**: Research findings are compressed while preserving exact source sentences for each citation. The URL context is provided in the source header for each webpage.

3. **Report Generation**: Draft and final reports generate text-fragment links using **range matching** format for reliable PDF rendering.

### Text Fragment Format (Range Matching)

Deep Research uses **range matching** to create short, reliable URLs that work in PDFs:

```
#:~:text=START_ANCHOR,END_ANCHOR
        ↑ first 3-5 words    ↑ last 3-5 words
                   ↑ comma separator (required)
```

**Why Range Matching:**
- Short URLs (under 100 chars) render correctly in PDFs
- Highlights the full sentence when clicked
- More reliable than encoding entire sentences

**Example:**
Source sentence: "Targeted therapies have reshaped the management of relapsed CLL."

```markdown
[1](https://example.com/article#:~:text=Targeted%20therapies%20have%20reshaped,management%20of%20relapsed%20CLL)
```

When clicked, this highlights the entire sentence.

### Dual-Format Citation System

Deep Research generates two citation formats:

**1. Markdown Output (for UI)** - Uses `<ref>` tags with verbatim source sentences:
```markdown
Targeted therapies changed treatment. <ref id="1">"Targeted therapies have reshaped the management of relapsed CLL."</ref>

### Sources
[1] Article Title: https://example.com/article
[2] Study Name: https://example.com/study
```

**2. PDF Output** - `<ref>` tags are converted to text-fragment URLs:
```markdown
Targeted therapies changed treatment [1](https://example.com/article#:~:text=Targeted%20therapies%20have%20reshaped,management%20of%20relapsed%20CLL)
```

**Why this approach:**
- `<ref>` format preserves exact source text for verification
- Citation check agent can verify quotes against research notes
- Downstream URL conversion creates clickable links for PDF
- Prevents hallucinated or paraphrased citations

### Anchor Selection Rules

- **START**: 3-5 words, must begin with capital letter
- **END**: 3-5 words, must end with complete word (not comma or parenthesis)
- **AVOID**: commas, brackets, parentheses, quotes, percent signs in selected text
- **Short sentences** (under 10 words): use 2-3 words for each anchor

### URL Encoding Rules

**Minimal encoding** - only encode spaces:
- Space → `%20`

**Avoid selecting text with:**
- Commas (would encode to `%2C`)
- Brackets (would encode to `%5B` `%5D`)
- Percent signs (would encode to `%25`)

### URL Conversion Process

Text-fragment URL conversion is handled by the LLM-based URL conversion agent (`convert_refs_to_urls` node) using structured prompts. The conversion follows the WICG Scroll-to-Text Fragment spec:

**Range Matching Format:**
```
#:~:text=START_ANCHOR,END_ANCHOR
```

**Conversion Rules (applied by LLM):**
- START: First 3-5 distinctive words (should begin with capital letter)
- END: Last 3-5 words (should end with complete word, exclude period)
- Encode spaces as `%20`
- Total fragment should be under 100 characters
- Avoid selecting text with commas, brackets, or special characters

**Fallback Behavior:**
If text-fragment generation would fail (special characters, length limits), the LLM uses plain URLs without fragments.

**Reference**: [WICG Scroll-to-Text Fragment Spec](https://wicg.github.io/scroll-to-text-fragment/)

## PDF Export

Final research reports are automatically converted to professional PDF format matching the OpenAI deep research aesthetic.

### Features

- **Professional Styling**: Serif fonts (Georgia, Times New Roman), 1-inch margins, proper heading hierarchy
- **Print-Friendly**: Optimized for both screen viewing and printing
- **Table Support**: Clean table formatting with alternating row colors
- **Code Blocks**: Syntax-highlighted code with proper formatting
- **Automatic Fallback**: If PDF generation fails or dependencies are missing, saves markdown to `/docs/` instead
- **Modular Design**: PDF generation utilities in `utils.py` for reusability

### Output Location

PDFs (or markdown fallback) are saved to:
```
/docs/{sanitized_title}_{timestamp}.pdf   # If WeasyPrint is installed
/docs/{sanitized_title}_{timestamp}.md    # Fallback if PDF fails
```

### System Dependencies

WeasyPrint requires native libraries for PDF generation:

**macOS**:
```bash
brew install cairo pango gdk-pixbuf libffi gobject-introspection
```

**Ubuntu/Debian**:
```bash
apt-get install libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info
```

**Note**: If system dependencies are missing, the system falls back to saving reports as markdown files.

### State Fields

The `AgentState` includes PDF-related fields:
- `pdf_path`: Path to the generated PDF file (or markdown fallback)
- `md_path`: Path to the markdown file with text-fragment URLs
- `final_report_pdf`: The report content with `<ref>` tags converted to text-fragment URLs (string)

### Workflow Integration

The PDF generation pipeline runs after final report generation with citation verification. The main graph uses a wrapper node to invoke a post-processing subgraph with isolated message state:

```
Main Graph:
final_report_generation → post_process_report → END

Post-Processing Subgraph (inside post_process_report):
check_citations → convert_refs_to_urls → convert_to_pdf
```

**Pipeline stages:**
1. **final_report_generation**: Generates report with `<ref id="N">"source sentence"</ref>` inline citations, saves notes to `/docs`
2. **post_process_report**: Wrapper node that invokes the post-processing subgraph with isolated state
   - **check_citations**: Verifies each ref's source sentence matches research notes exactly (uses `citation_think_tool` via `create_agent`)
   - **convert_refs_to_urls**: LLM converts `<ref>` tags to text-fragment URL links using range matching
   - **convert_to_pdf**: Generates PDF from URL-converted markdown, saves markdown file, adds report to UI messages


