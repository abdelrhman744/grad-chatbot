"""
schemas.py

Pydantic schemas used by the ReAct agent.
"""

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


# =====================================================
# Tool Names
# =====================================================

class ToolName(str, Enum):

    RETRIEVE = "retrieve"

    GENERATE = "generate"

    SUMMARIZE = "summarize"

    COMPARE = "compare"

    RESPOND = "respond"


# =====================================================
# Tools that end the ReAct loop.
#
# Anything not in this set (currently just RETRIEVE) must be followed
# by another action instead of being treated as a final answer.
# =====================================================

TERMINAL_TOOLS = frozenset(
    {
        ToolName.GENERATE,
        ToolName.SUMMARIZE,
        ToolName.COMPARE,
        ToolName.RESPOND,
    }
)


# =====================================================
# Tool Arguments
# =====================================================

class RetrieveArguments(BaseModel):

    question: str

    top_k: int = 5


class GenerateArguments(BaseModel):

    question: str


class CompareArguments(BaseModel):

    question: str


class SummarizeArguments(BaseModel):
    pass


class RespondArguments(BaseModel):

    question: str


# =====================================================
# Base Action
# =====================================================

class BaseAction(BaseModel):

    thought: str

    action: ToolName


# =====================================================
# Retrieve
# =====================================================

class RetrieveAction(BaseAction):

    action: Literal[ToolName.RETRIEVE]

    arguments: RetrieveArguments


# =====================================================
# Generate
# =====================================================

class GenerateAction(BaseAction):

    action: Literal[ToolName.GENERATE]

    arguments: GenerateArguments


# =====================================================
# Compare
# =====================================================

class CompareAction(BaseAction):

    action: Literal[ToolName.COMPARE]

    arguments: CompareArguments


# =====================================================
# Summarize
# =====================================================

class SummarizeAction(BaseAction):

    action: Literal[ToolName.SUMMARIZE]

    arguments: SummarizeArguments


# =====================================================
# Respond
#
# Used for greetings, small talk, and follow-ups that can be answered
# from conversation memory alone, without hitting the vector store.
# =====================================================

class RespondAction(BaseAction):

    action: Literal[ToolName.RESPOND]

    arguments: RespondArguments


# =====================================================
# Agent Action
# =====================================================

AgentAction = Annotated[
    Union[
        RetrieveAction,
        GenerateAction,
        CompareAction,
        SummarizeAction,
        RespondAction,
    ],
    Field(discriminator="action"),
]


# =====================================================
# Execution Context
# =====================================================

class ExecutionContext(BaseModel):

    # The original user question for this turn
    question: str = ""

    # Detected language of the question ("en" / "ar")
    language: str = "en"

    # Retrieved chunks
    documents: list = Field(default_factory=list)

    # Tool observations
    observations: list[dict] = Field(default_factory=list)

    # Questions already retrieved (avoids redundant retrieval calls)
    retrieved_questions: list[str] = Field(default_factory=list)

    # ReAct reasoning history
    scratchpad: list[dict] = Field(default_factory=list)

    # Rolling conversation summary, injected at the start of the turn
    memory: str = ""

    summary: str | None = None

    comparison: str | None = None

    answer: str | None = None

    # Source documents formatted for display, filled in by generate/respond
    sources: str | None = None

    def has_final_answer(self) -> bool:
        return bool(self.answer or self.summary or self.comparison)

    def final_text(self) -> str:
        return self.answer or self.summary or self.comparison or ""
