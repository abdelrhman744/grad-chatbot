"""
registry.py

Builds the tool registry used by the ReAct agent. A factory function
(rather than module-level singletons) because generate/respond need a
`memory_text_provider` callback injected by the owning Agent instance.
"""

from __future__ import annotations

from typing import Callable

from .tools.retrieve_tool import RetrieveTool
from .tools.generate_tool import GenerateTool
from .tools.summarize_tool import SummarizeTool
from .tools.compare_tool import CompareTool
from .tools.respond_tool import RespondTool


def build_tools(memory_text_provider: Callable[[], str]) -> dict:
    return {
        "retrieve": RetrieveTool(),
        "generate": GenerateTool(memory_text_provider=memory_text_provider),
        "summarize": SummarizeTool(),
        "compare": CompareTool(),
        "respond": RespondTool(memory_text_provider=memory_text_provider),
    }
