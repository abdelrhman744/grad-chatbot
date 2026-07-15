"""
schemas.py

Pydantic schemas used by the ReAct agent: the set of actions it can
choose between at each step, and the execution context threaded through
the reasoning loop.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


# ── Tool Names ───────────────────────────────────────────────────────────────

class ToolName(str, Enum):
    RETRIEVE = "retrieve"
    GENERATE = "generate"
    SUMMARIZE = "summarize"
    COMPARE = "compare"
    RESPOND = "respond"


TERMINAL_TOOLS = frozenset(
    {ToolName.GENERATE, ToolName.SUMMARIZE, ToolName.COMPARE, ToolName.RESPOND}
)


# ── Tool Arguments ───────────────────────────────────────────────────────────

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


# ── Actions ──────────────────────────────────────────────────────────────────

class BaseAction(BaseModel):
    thought: str
    action: ToolName


class RetrieveAction(BaseAction):
    action: Literal[ToolName.RETRIEVE]
    arguments: RetrieveArguments


class GenerateAction(BaseAction):
    action: Literal[ToolName.GENERATE]
    arguments: GenerateArguments


class CompareAction(BaseAction):
    action: Literal[ToolName.COMPARE]
    arguments: CompareArguments


class SummarizeAction(BaseAction):
    action: Literal[ToolName.SUMMARIZE]
    arguments: SummarizeArguments


class RespondAction(BaseAction):
    action: Literal[ToolName.RESPOND]
    arguments: RespondArguments


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


# ── Execution Context ────────────────────────────────────────────────────────

class ExecutionContext(BaseModel):
    # Retrieved chunks, as plain dicts: {"id", "text", "metadata": {...}}
    documents: list = Field(default_factory=list)

    # Tool observations, fed back into the next reasoning step
    observations: list[dict] = Field(default_factory=list)

    # Questions already retrieved this turn (duplicate-retrieval guard)
    retrieved_questions: list[str] = Field(default_factory=list)

    summary: str | None = None
    comparison: str | None = None
    answer: str | None = None

    # Detected language for this turn ("ar" / "en")
    language: str = "en"

    def final_answer(self) -> str | None:
        """Whichever terminal output was produced this turn, if any."""
        return self.answer or self.summary or self.comparison
