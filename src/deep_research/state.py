from typing import Annotated, Optional
from pydantic import BaseModel, Field
import operator
from langgraph.graph import MessagesState
from langchain_core.messages import MessageLikeRepresentation
from typing_extensions import TypedDict

###################
# Structured Outputs
###################
class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )

class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""

class ClaimSourcePair(BaseModel):
    """Atomic claim with its supporting source sentence for text-fragment link generation."""
    claim: str = Field(
        description="A single factual claim extracted from the webpage"
    )
    source_sentence: str = Field(
        description="The exact verbatim sentence from the source that supports this claim - do NOT paraphrase"
    )


class Summary(BaseModel):
    """Structured summary output with atomic claim-source pairs for citation generation."""
    summary: str = Field(
        description="A comprehensive summary of the webpage content"
    )
    claim_source_pairs: list[ClaimSourcePair] = Field(
        description="List of atomic claim-source pairs extracted from the webpage",
        default_factory=list
    )

class ClarifyWithUser(BaseModel):
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )

class DraftReport(BaseModel):
    draft_report: str = Field(
        description="A draft report that will be used to guide the research.",
    )


###################
# State Definitions
###################

def override_reducer(current_value, new_value):
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
    
class AgentInputState(MessagesState):
    """InputState is only 'messages'"""

class AgentState(MessagesState):
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    draft_report: str
    final_report: str
    final_report_pdf: str = ""  # Report with refs converted to text-fragment URLs for PDF generation
    brief_refinement_rounds: int = 0
    pdf_path: Optional[str] = None
    md_path: Optional[str] = None  # Markdown output for observing URL link patterns

class SupervisorState(TypedDict):
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []
    draft_report: str

class ResearcherState(TypedDict):
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherOutputState(BaseModel):
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []