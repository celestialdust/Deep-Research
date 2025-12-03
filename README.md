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
- **Enable Human-in-the-Loop** (default: true): Whether to enable human approval for the research brief before proceeding to draft report generation
- **Max Brief Refinement Rounds** (default: 1): Maximum number of times the research brief can be refined based on human feedback before proceeding automatically

#### Research Configuration

- **Search API** (default: Tavily): Choose from Tavily (works with all models), OpenAI Native Web Search, Anthropic Native Web Search, or None
- **Max Researcher Iterations** (default: 3): Number of times the Research Supervisor will reflect on research and ask follow-up questions
- **Max React Tool Calls** (default: 5): Maximum number of tool calling iterations in a single researcher step
- **Enable Search for Brief** (default: true): Whether to enable Tavily search for the research brief generation node (max 1 search allowed)
- **Enable Search for Draft** (default: true): Whether to enable Tavily search for the draft report generation node (max 1 search allowed)

#### Model Configuration

- **Supported Models**: GPT-5, O3, O4-mini and other Azure OpenAI models
- **GPT-5 Configuration**: 
  - Deployment name: `gsds-gpt-5` (default)
  - API Version: `2025-04-01-preview`
  - Token limit: 200,000 tokens
- **Model Selection**: Configure different models for summarization, research, compression, final report generation, citation checking, and PDF conversion
- **Azure OpenAI Settings**: Customize endpoint, API version, and deployment names for each model

#### Citation & PDF Configuration

- **Citation Check Model** (default: azure_openai:o4-mini): Model for verifying citations match source text exactly
- **Citation Check Model Max Tokens** (default: 16,000): Maximum output tokens for citation check model
- **PDF Conversion Model** (default: azure_openai:o4-mini): Model for converting `<ref>` tags to text-fragment URLs
- **PDF Conversion Model Max Tokens** (default: 16,000): Maximum output tokens for PDF conversion model

#### Context Engineering Configuration

- **Enable Context Summarization** (default: true): Whether to enable automatic context summarization when approaching token limits
- **Max Tokens Before Summary** (default: 100,000): Token threshold to trigger context summarization. When messages exceed this limit, older messages will be summarized
- **Messages to Keep** (default: 20): Number of recent messages to preserve after summarization
- **Context Summarization Model** (default: azure_openai:o4-mini): Model to use for context summarization (o4-mini recommended for speed)

## Architecture Overview

Deep Research Agent uses a sophisticated multi-agent architecture built on LangGraph with the following components:

- **User Clarification**: Optionally asks clarifying questions before starting research
- **Research Brief Generator**: Creates a detailed research brief from user messages, with optional Tavily search for up-to-date context
- **Human-in-the-Loop Approval**: Allows human review and refinement of the research brief before proceeding (optional, configurable)
- **Draft Report Generator**: Creates an initial draft report from the research brief that serves as a baseline for iterative refinement, with optional Tavily search for preliminary context
- **Research Supervisor**: Orchestrates the research process using a diffusion algorithm, breaks down complex queries, and manages parallel research units with strategic reflection via `think_tool`
- **Research Agents**: Conduct focused research on specific topics using available tools and search APIs
- **Compression Agent**: Synthesizes and compresses research findings from multiple agents
- **Draft Report Refiner**: Iteratively refines the draft report based on new research findings using the diffusion algorithm
- **Report Generator**: Creates comprehensive final reports from all research findings with enhanced insightfulness and helpfulness criteria
- **Citation Check Agent**: Verifies that all `<ref>` citations match research notes exactly, correcting paraphrased or hallucinated source sentences
- **URL Conversion Agent**: Converts `<ref id="N">"source"</ref>` tags to text-fragment URLs using range matching for PDF generation

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

- **`deep_researcher.py`**: Contains the main LangGraph workflow implementation with nodes for:
  - User clarification
  - Research brief generation (with optional Tavily search)
  - Human-in-the-loop research brief approval
  - Draft report generation (with optional Tavily search and `<ref>` citations)
  - Research supervisor (with diffusion algorithm and context summarization)
  - Parallel research execution (with context summarization)
  - Research compression (with exact source sentence preservation)
  - Draft report refinement
  - Final report generation (with `<ref>` inline citations and async note-saving to `/docs`)
  - Citation check (verifies citations match research notes exactly using `create_agent`)
  - URL conversion (converts `<ref>` tags to text-fragment URLs for PDF)
  - PDF conversion (markdown → HTML → PDF with WeasyPrint)

- **`state.py`**: Defines all state classes and data models:
  - `AgentState`: Main workflow state (includes `draft_report`, `brief_refinement_rounds`, `pdf_path`, `final_report_pdf` fields)
  - `SupervisorState`: Research supervisor state (includes `draft_report` field)
  - `ResearcherState`: Individual researcher state
  - `ClaimSourcePair`: Atomic claim with exact source sentence for text-fragment citations (claim + source_sentence fields)
  - `Summary`: Structured webpage summary with claim-source pairs where every key claim is backed by a source sentence
  - `DraftReport`: Structured output for draft report generation
  - Structured outputs for research coordination

- **`utils.py`**: Utility functions including:
  - `MessageSummarizer`: Context window summarization class (mirrors LangChain v1's `SummarizationMiddleware`)
  - `think_tool`: Strategic reflection tool for research planning
  - `citation_think_tool`: Analysis tool for verifying citation accuracy against research notes
  - `refine_draft_report`: Tool for iteratively refining the draft report
  - `tavily_search`: Tavily search tool for research brief and draft generation
  - `summarize_webpage`: Webpage summarization with atomic claim-source pair extraction (with automatic retry on token limits)
  - PDF generation utilities: `generate_pdf_from_markdown`, `save_markdown_file`, `extract_title_from_markdown`, `sanitize_filename`
  - `PDF_CSS_STYLES`: Professional CSS styling for PDF reports
  - URL utilities with robust fallback handling:
    - `create_citation_link`: One-step citation link from URL + source sentence
    - `create_text_fragment_url`: Create range-matched text fragment URLs
    - `validate_url`: Validate and sanitize URLs with automatic https:// prefix
    - `normalize_url`: Clean up common URL issues (double slashes, trailing hashes)
    - `extract_text_fragment`: Parse text fragment components from URLs
    - `sanitize_anchor_text`: Clean anchor text, remove problematic chars
    - `get_anchor_from_sentence`: Extract optimal start/end anchors with capital-letter detection
    - `encode_text_fragment_anchor` / `decode_text_fragment_anchor`: URL encoding/decoding
    - `has_problematic_chars`: Check for characters that break URLs/PDFs
  - MCP tool loading and authentication
  - Search API integration (Tavily, OpenAI, Anthropic)
  - Model configuration builders with GPT-5 support
  - Token limit management with extended model coverage

- **`prompts.py`**: System prompts and templates for:
  - Research clarification
  - Research brief generation (with optional search tool instructions)
  - Research planning with diffusion algorithm
  - Draft report generation (with `<ref id="N">"source"</ref>` citation format)
  - Research execution with strategic reflection
  - Webpage summarization with atomic claim-source pair extraction (every key claim must be backed by a source sentence)
  - Content summarization and compression (with exact source sentence preservation)
  - Draft report refinement
  - Final report generation (with `<ref>` inline citations and Sources section rules)
  - Citation verification (for checking citations against research notes)
  - URL conversion (for converting refs to text-fragment URLs)

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
- **Both sync and async support**: `summarize_if_needed()` (async) and `summarize_if_needed_sync()` (sync)

**Applied to nodes:**
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
   
   **Selective Extraction**: Only 3-7 most citation-worthy claims get source sentences (headline facts, primary findings, unique insights). The summary remains comprehensive while claim-source pairs focus on the most important facts likely to be cited in final reports.
   
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

### URL Utilities

The `utils.py` module provides robust URL handling with comprehensive fallback:

```python
from deep_research.utils import (
    create_text_fragment_url,    # Create range-matched text fragment URLs
    create_citation_link,        # One-step citation link from URL + sentence
    validate_url,                # Validate and sanitize URLs
    normalize_url,               # Clean up common URL issues
    extract_text_fragment,       # Parse text fragment from URL
    sanitize_anchor_text,        # Clean anchor text for URL encoding
    get_anchor_from_sentence,    # Extract start/end anchors from sentence
    encode_text_fragment_anchor, # Encode text for URL fragment
    decode_text_fragment_anchor, # Decode URL fragment to text
    has_problematic_chars        # Check for chars that break URLs/PDFs
)

# Create a citation link from URL and source sentence (recommended)
url = create_citation_link(
    base_url="https://example.com/article",
    source_sentence="Targeted therapies have reshaped the management of relapsed CLL."
)
# Returns: https://example.com/article#:~:text=Targeted%20therapies%20have%20reshaped,management%20of%20relapsed%20CLL

# Or manually create with specific anchors
url = create_text_fragment_url(
    base_url="https://example.com/article",
    start_anchor="Targeted therapies have reshaped",
    end_anchor="management of relapsed CLL"
)

# Validate and normalize URLs
is_valid, result = validate_url("example.com/page")
# is_valid: True, result: "https://example.com/page"

# Extract anchors from a sentence
start = get_anchor_from_sentence("The study enrolled 1847 participants.", position="start")
end = get_anchor_from_sentence("The study enrolled 1847 participants.", position="end")
# start: "The study enrolled 1847"
# end: "enrolled 1847 participants"
```

**Robustness Features:**
- Automatic URL scheme detection and addition (`https://`)
- URL normalization (removes double slashes, trailing hashes)
- Anchor length limits (max 100 chars for PDF compatibility)
- Automatic fallback to plain URL if fragment creation fails
- Detection of problematic characters (commas, brackets, etc.)
- Smart anchor extraction that finds capital-letter starts
- Handles short sentences with fewer anchor words

### Fallback Behavior

If text-fragment generation fails, the system falls back to plain URLs:
- Sentence has many special characters → plain URL
- URL would exceed 100 characters → plain URL
- Encoding error → plain URL

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

The PDF generation pipeline runs after final report generation with citation verification:
```
final_report_generation → check_citations → convert_refs_to_urls → convert_to_pdf → END
```

**Pipeline stages:**
1. **final_report_generation**: Generates report with `<ref id="N">"source sentence"</ref>` inline citations
2. **check_citations**: Verifies each ref's source sentence matches research notes exactly (uses `citation_think_tool`)
3. **convert_refs_to_urls**: LLM converts `<ref>` tags to text-fragment URL links using range matching
4. **convert_to_pdf**: Generates PDF from URL-converted markdown, adds report to UI messages

## Future Developments

### Engineering

- Try different search apis and see which one works best
- Reinforcement learning for better dynamic reasoning/planning + tool use
- Web Search Resource validation agent
- ~~Apply Context Engineering practices~~ (Completed)
- ~~Retraceable Citations with Text-Fragment Links~~ (Completed)
- ~~PDF Export for Professional Reports~~ (Completed)
- ~~Citation Verification Agent~~ (Completed)
- Integrate domain specific insights from feedbacks

### Product/UX

- Create a separate UI for deep research while putting it as a add-on in clinical trial protocol copilot
- Send deep research reports as attachments to users in email after deep research is done


