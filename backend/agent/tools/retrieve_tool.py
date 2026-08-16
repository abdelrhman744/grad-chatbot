"""
retrieve_tool.py

Tool for retrieving relevant document chunks from the vector database.
Wraps services.rag_service.retrieve() so the agent never touches
langchain/Qdrant directly.
"""

from __future__ import annotations

from agent.schemas import ExecutionContext
from services import rag_service
from utils import timing


class RetrieveTool:
    name = "retrieve"

    def __init__(self, conversation_id: str):
        # Required — every retrieval must be scoped to the owning
        # conversation's own documents (see Document Isolation). Injected
        # by agent/registry.py::build_tools, sourced from Agent.conversation_id.
        self.conversation_id = conversation_id

    def run(
        self, context: ExecutionContext, question: str, top_k: int = 5, raw_question: str = ""
    ) -> ExecutionContext:
        # A single aggregated "how long did this retrieve call take"
        # substage, wrapping the whole rag_service.retrieve() call (query
        # variants + embedding + Qdrant + reranking + MMR together) — see
        # utils/timing.py's substage() docstring: this deliberately does
        # NOT use stage() here, since retrieve() already contains several
        # of its own top-level stages internally, and a stage()-within-
        # stage() wrapper would double-count wall-clock time in
        # RequestTimer.report()'s "unaccounted" total. substage() avoids
        # that (it only adds to the cumulative `notes` bucket, not the
        # summed `stages` list). For a multi-question turn this is called
        # more than once and correctly SUMS into one "retrieval_total_ms"
        # figure for the whole turn.
        with timing.substage("retrieval_total_ms"):
            new_documents = rag_service.retrieve(
                question, self.conversation_id, lang=context.language, top_k=top_k,
                raw_question=raw_question,
            )

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
