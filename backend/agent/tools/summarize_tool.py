"""
summarize_tool.py

Terminal tool: summarizes the documents retrieved so far.
"""

from __future__ import annotations

from agent.schemas import ExecutionContext
from services import rag_service


class SummarizeTool:
    name = "summarize"

    def run(self, context: ExecutionContext) -> ExecutionContext:
        if not context.documents:
            context.summary = (
                "لا توجد مستندات لتلخيصها." if context.language == "ar"
                else "No documents found to summarize."
            )
            return context

        context.summary = rag_service.summarize(context.documents, lang=context.language)
        context.observations.append({"tool": "summarize", "status": "summarized"})

        return context
