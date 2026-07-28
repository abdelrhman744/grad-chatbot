"""
summarize_tool.py

Summarize retrieved documents.
"""

from __future__ import annotations

from rag.agent.schemas import ExecutionContext
from rag.generator import Generator

NO_DOCUMENTS = {
    "en": "No documents were found to summarize.",
    "ar": "لا توجد مستندات لتلخيصها.",
}

PROMPT_TEMPLATE = {
    "en": "Summarize the following documents clearly and concisely:\n\n{documents}",
    "ar": "لخّص المستندات التالية بإيجاز ووضوح:\n\n{documents}",
}


class SummarizeTool:

    def __init__(self, generator: Generator | None = None):
        self.generator = generator or Generator()

    # =====================================================
    # Public Method
    # =====================================================

    def run(self, context: ExecutionContext) -> ExecutionContext:

        language = context.language

        if not context.documents:
            context.summary = NO_DOCUMENTS.get(language, NO_DOCUMENTS["en"])
            context.observations.append({"tool": "summarize", "status": "no_documents"})
            return context

        document_text = "\n\n".join(document["text"] for document in context.documents)

        prompt = PROMPT_TEMPLATE.get(language, PROMPT_TEMPLATE["en"]).format(documents=document_text)

        context.summary = self.generator.generate(prompt)

        context.observations.append({"tool": "summarize", "status": "summarized"})

        return context
