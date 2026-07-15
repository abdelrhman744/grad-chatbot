"""
respond_tool.py

Terminal tool: answers directly from conversation memory without
querying the vector database. Used for greetings, small talk, and
follow-up questions the agent judges answerable from memory alone.
"""

from __future__ import annotations

from agent.schemas import ExecutionContext
from services import rag_service


class RespondTool:
    name = "respond"

    def __init__(self, memory_text_provider=None):
        self._memory_text_provider = memory_text_provider or (lambda: "")

    def run(self, context: ExecutionContext, question: str) -> ExecutionContext:
        memory_text = self._memory_text_provider()
        context.answer = rag_service.answer_from_memory(question, memory_text, lang=context.language)
        context.observations.append({"tool": "respond", "status": "answered_from_memory"})
        return context
