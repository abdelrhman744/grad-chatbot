"""
compare_tool.py

Terminal tool: compares information across the documents retrieved so far.
"""

from __future__ import annotations

from agent.schemas import ExecutionContext
from services import rag_service


class CompareTool:
    name = "compare"

    def run(self, context: ExecutionContext, question: str) -> ExecutionContext:
        if not context.documents:
            context.comparison = (
                "لا توجد مستندات كافية للمقارنة." if context.language == "ar"
                else "No documents found to compare."
            )
            return context

        context.comparison = rag_service.compare(question, context.documents, lang=context.language)
        context.observations.append({"tool": "compare", "status": "compared"})

        return context
