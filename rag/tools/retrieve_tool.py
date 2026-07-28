"""
retrieve_tool.py

Tool for retrieving relevant documents.

Note: this only wraps the retrieval call with the chatbot-layer
bookkeeping (dedup, observations). The retrieval logic itself
(Retriever/vector search) is out of scope and untouched here.

`run()` is split into `fetch()` (pure retrieval — no shared state,
safe to call concurrently from multiple threads) and `merge()`
(mutates the shared ExecutionContext — must run sequentially). This
split exists so compound/multi-question turns can fire off several
`fetch()` calls in parallel (see Agent._retrieve_compound_question)
and then merge each result into context one at a time afterward,
without risking a race on context.documents/context.observations.
"""

from __future__ import annotations

import logging

from ..agent.schemas import ExecutionContext
from ..retriever import Retriever

logger = logging.getLogger(__name__)


class RetrieveTool:

    def __init__(self):
        self.retriever = Retriever()

    # =====================================================
    # Public Method
    # =====================================================

    def run(
        self,
        context: ExecutionContext,
        question: str,
        top_k: int = 5,
    ) -> ExecutionContext:

        new_documents, error = self.fetch(question, top_k)

        return self.merge(context, question, new_documents, error)

    # =====================================================
    # Parallel-safe fetch (no context access)
    # =====================================================

    def fetch(
        self,
        question: str,
        top_k: int = 5,
    ) -> tuple[list | None, Exception | None]:
        """
        Run just the retrieval computation and return (documents,
        error) instead of raising/mutating context. Touches nothing
        shared, so multiple calls can safely run concurrently (e.g.
        one per sub-question of a compound message) in a thread pool.
        """

        try:
            return self.retriever.retrieve(question=question, top_k=top_k), None
        except Exception as error:
            logger.warning("[RetrieveTool] retrieval failed for %r: %s", question, error)
            return None, error

    # =====================================================
    # Sequential merge (mutates shared context)
    # =====================================================

    def merge(
        self,
        context: ExecutionContext,
        question: str,
        new_documents: list | None,
        error: Exception | None = None,
    ) -> ExecutionContext:
        """
        Fold a fetch() result into the shared ExecutionContext. Must be
        called sequentially (never from multiple threads at once) since
        it mutates context.documents / context.observations in place.
        """

        if error is not None or new_documents is None:
            context.observations.append(
                {
                    "tool": "retrieve",
                    "question": question,
                    "chunks": 0,
                    "documents": [],
                    "status": "error",
                }
            )
            return context

        existing_ids = {document["id"] for document in context.documents}

        added = 0
        titles: list[str] = []

        for document in new_documents:
            if document["id"] not in existing_ids:
                context.documents.append(document)
                existing_ids.add(document["id"])
                added += 1

            title = document.get("metadata", {}).get("title", "Unknown")
            if title not in titles:
                titles.append(title)

        context.observations.append(
            {
                "tool": "retrieve",
                "question": question,
                "chunks": added,
                "documents": titles,
            }
        )

        return context
