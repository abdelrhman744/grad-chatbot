"""
generate_tool.py

Terminal tool: produces the final answer from the documents the agent has
retrieved so far, enriched with conversation memory. Falls back to a
memory-only answer if no documents were retrieved but memory exists.
"""

from __future__ import annotations

from agent.schemas import ExecutionContext
from services import rag_service


class GenerateTool:
    name = "generate"

    def __init__(self, memory_text_provider=None):
        # Injected by the Agent so this tool can read conversation memory
        # without owning a MemoryManager itself.
        self._memory_text_provider = memory_text_provider or (lambda: "")

    def run(self, context: ExecutionContext, question: str) -> ExecutionContext:
        memory_text = self._memory_text_provider()

        if not context.documents:
            if memory_text:
                context.answer = rag_service.answer_from_memory(
                    question, memory_text, lang=context.language
                )
            else:
                context.answer = (
                    "لا توجد مستندات كافية للإجابة على هذا السؤال."
                    if context.language == "ar"
                    else "No relevant documents were found to answer this question."
                )
            return context

        context.answer = rag_service.generate_answer(
            question, context.documents, lang=context.language, memory=memory_text
        )

        context.observations.append({"tool": "generate", "status": "answer_generated"})

        return context
