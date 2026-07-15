"""
retrieve_tool.py

Tool for retrieving relevant document chunks from the vector database.
Wraps services.rag_service.retrieve() so the agent never touches
langchain/Qdrant directly.
"""

from __future__ import annotations

from agent.schemas import ExecutionContext
from services import rag_service


class RetrieveTool:
    name = "retrieve"

    def run(self, context: ExecutionContext, question: str, top_k: int = 5) -> ExecutionContext:
        new_documents = rag_service.retrieve(question, lang=context.language, top_k=top_k)

        existing_ids = {doc["id"] for doc in context.documents}

        added = 0
        titles: list[str] = []

        for document in new_documents:
            if document["id"] not in existing_ids:
                context.documents.append(document)
                existing_ids.add(document["id"])
                added += 1

            title = document["metadata"].get("title", "Unknown")
            if title not in titles:
                titles.append(title)

        context.observations.append({
            "tool": "retrieve",
            "question": question,
            "chunks_added": added,
            "total_documents": len(context.documents),
            "sources": titles,
        })

        return context
