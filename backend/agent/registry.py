"""
registry.py

Builds the tool registry used by the ReAct agent. A factory function
(rather than module-level singletons) because generate/respond need a
`memory_text_provider` callback injected by the owning Agent instance.
"""

from __future__ import annotations

from typing import Callable, Optional

from .tools.retrieve_tool import RetrieveTool
from .tools.generate_tool import GenerateTool
from .tools.summarize_tool import SummarizeTool
from .tools.compare_tool import CompareTool
from .tools.respond_tool import RespondTool
from .tools.report_tool import ReportTool


def build_tools(
    memory_text_provider: Callable[[], str],
    conversation_id: str,
    active_document_provider: Optional[Callable[[], Optional[str]]] = None,
    active_document_setter: Optional[Callable[[str], None]] = None,
) -> dict:
    return {
        "retrieve": RetrieveTool(conversation_id=conversation_id),
        "generate": GenerateTool(memory_text_provider=memory_text_provider),
        "summarize": SummarizeTool(),
        "compare": CompareTool(),
        "respond": RespondTool(memory_text_provider=memory_text_provider),
        "report": ReportTool(
            conversation_id=conversation_id,
            active_document_provider=active_document_provider,
            active_document_setter=active_document_setter,
        ),
    }
