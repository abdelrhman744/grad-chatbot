"""
registry.py

Builds the tool registry used by the ReAct agent.

A factory function (rather than module-level singletons) because
generate/respond need a `memory_text_provider` callback injected by the
owning Agent instance, so each tool can fall back to conversation
memory when there's nothing useful in the retrieved documents.
"""

from __future__ import annotations

from typing import Callable

from rag.generator import Generator
from rag.tools.retrieve_tool import RetrieveTool
from rag.tools.generate_tool import GenerateTool
from rag.tools.summarize_tool import SummarizeTool
from rag.tools.compare_tool import CompareTool
from rag.tools.respond_tool import RespondTool


def build_tools(
    memory_text_provider: Callable[[], str],
    last_assistant_provider: Callable[[], str] | None = None,
) -> dict:
    # One shared Generator (and therefore one shared model/token/top_p
    # config) instead of each tool instantiating its own.
    generator = Generator()

    return {
        "retrieve": RetrieveTool(),
        "generate": GenerateTool(generator=generator, memory_text_provider=memory_text_provider),
        "summarize": SummarizeTool(generator=generator),
        "compare": CompareTool(generator=generator),
        "respond": RespondTool(
            generator=generator,
            memory_text_provider=memory_text_provider,
            last_assistant_provider=last_assistant_provider,
        ),
    }