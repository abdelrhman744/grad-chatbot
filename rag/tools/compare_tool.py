"""
compare_tool.py

Compare topics using retrieved documents.
"""

from __future__ import annotations

from rag.agent.schemas import ExecutionContext
from rag.generator import Generator

NO_DOCUMENTS = {
    "en": "No documents were found to compare.",
    "ar": "لا توجد مستندات كافية للمقارنة.",
}

PROMPT_TEMPLATE = {
    "en": "Using ONLY the provided documents, answer the following comparison request.\n\nQuestion:\n{question}\n\nDocuments:\n\n{documents}",
    "ar": "باستخدام المستندات التالية فقط، أجب عن طلب المقارنة:\n\nالسؤال:\n{question}\n\nالمستندات:\n\n{documents}",
}


class CompareTool:

    def __init__(self, generator: Generator | None = None):
        self.generator = generator or Generator()

    # =====================================================
    # Public Method
    # =====================================================

    def run(self, context: ExecutionContext, question: str) -> ExecutionContext:

        language = context.language

        if not context.documents:
            context.comparison = NO_DOCUMENTS.get(language, NO_DOCUMENTS["en"])
            context.observations.append({"tool": "compare", "status": "no_documents"})
            return context

        document_text = "\n\n".join(document["text"] for document in context.documents)

        prompt = PROMPT_TEMPLATE.get(language, PROMPT_TEMPLATE["en"]).format(
            question=question, documents=document_text
        )

        context.comparison = self.generator.generate(prompt)

        context.observations.append({"tool": "compare", "status": "compared"})

        return context
