"""
generate_tool.py

Terminal tool: produces the final answer from the documents the agent
has retrieved so far, enriched with conversation memory. Falls back to
a memory-only answer if no documents were retrieved but memory exists.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Iterator

from rag import prompt as prompt_lib
from rag.agent.schemas import ExecutionContext
from rag.generator import Generator

# context.documents can accumulate chunks from several retrieve calls
# in one turn (e.g. one per sub-question of a compound message), each
# already capped at its own top_k but not capped in total. A bigger
# prompt takes longer to process even on fast inference, and beyond a
# certain point extra chunks mostly add noise rather than signal since
# the most relevant ones are already ranked first. Cap the total chunk
# count sent to the generator to the highest-scoring N.
MAX_CONTEXT_CHUNKS = int(os.getenv("RAG_MAX_CONTEXT_CHUNKS", "6"))


class GenerateTool:

    def __init__(self, generator: Generator | None = None, memory_text_provider=None):
        self.generator = generator or Generator()

        # Injected by the Agent so this tool can read conversation
        # memory without owning a MemoryManager itself.
        self._memory_text_provider = memory_text_provider or (lambda: "")

    # =====================================================
    # Public Method
    # =====================================================

    def run(self, context: ExecutionContext, question: str) -> ExecutionContext:

        memory_text = self._memory_text_provider()
        language = context.language

        if not context.documents:
            if memory_text:
                context.answer = self._answer_from_memory(question, memory_text, language)
            else:
                context.answer = prompt_lib.NO_ANSWER.get(language, prompt_lib.NO_ANSWER["en"])

            context.observations.append({"tool": "generate", "status": "answered_from_memory"})
            return context

        selected_documents = self._select_top_chunks(context.documents)
        document_context = self._build_grouped_context(selected_documents)

        answer_prompt = prompt_lib.build_prompt_with_memory(
            document_context, question, language, memory_text
        )

        raw_answer = self.generator.generate(answer_prompt)
        context.answer = prompt_lib.clean_answer(raw_answer, language)
        context.sources = prompt_lib.build_sources(selected_documents, language)

        context.observations.append({"tool": "generate", "status": "answer_generated"})

        return context

    # =====================================================
    # Streaming variant
    # =====================================================

    def stream_run(self, context: ExecutionContext, question: str) -> Iterator[str]:
        """
        Same behavior as `run`, but yields the answer as it's generated
        instead of returning only once the full text is ready. Still
        finishes by writing `context.answer` / `context.sources`, so
        callers can treat the context the same way once the generator
        is exhausted (memory, sources display, etc.).
        """
        memory_text = self._memory_text_provider()
        language = context.language

        if not context.documents:
            if memory_text:
                prompt = prompt_lib.build_memory_prompt(question, memory_text, language)
            else:
                context.answer = prompt_lib.NO_ANSWER.get(language, prompt_lib.NO_ANSWER["en"])
                context.observations.append({"tool": "generate", "status": "answered_from_memory"})
                yield context.answer
                return

            context.observations.append({"tool": "generate", "status": "answered_from_memory"})
        else:
            selected_documents = self._select_top_chunks(context.documents)
            document_context = self._build_grouped_context(selected_documents)
            prompt = prompt_lib.build_prompt_with_memory(
                document_context, question, language, memory_text
            )
            context.sources = prompt_lib.build_sources(selected_documents, language)
            context.observations.append({"tool": "generate", "status": "answer_generated"})

        collected = []
        for delta in self.generator.stream(prompt):
            collected.append(delta)
            yield delta

        context.answer = prompt_lib.clean_answer("".join(collected), language)

    # =====================================================
    # Internal
    # =====================================================

    def _answer_from_memory(self, question: str, memory_text: str, language: str) -> str:
        prompt = prompt_lib.build_memory_prompt(question, memory_text, language)
        return prompt_lib.clean_answer(self.generator.generate(prompt), language)

    @staticmethod
    def _select_top_chunks(documents: list[dict], max_chunks: int = MAX_CONTEXT_CHUNKS) -> list[dict]:
        """
        Cap the chunks passed to the generator to the highest-scoring
        `max_chunks`. Scores by rerank_score when present (set by the
        cross-encoder reranker), falling back to rrf_score (set when
        reranking was skipped for an already-small candidate set — see
        retriever.py), then 0.0 if neither is present. No-op if there
        are already fewer chunks than the cap.
        """

        if len(documents) <= max_chunks:
            return documents

        def score(document: dict) -> float:
            return document.get("rerank_score", document.get("rrf_score", 0.0))

        return sorted(documents, key=score, reverse=True)[:max_chunks]

    @staticmethod
    def _build_grouped_context(documents: list[dict]) -> str:
        """Group retrieved chunks by document_id so the model sees each
        source document's chunks together instead of interleaved."""

        grouped_documents = defaultdict(list)

        for chunk in documents:
            document_id = chunk.get("metadata", {}).get("document_id", "unknown")
            grouped_documents[document_id].append(chunk)

        parts = []

        for chunks in grouped_documents.values():
            metadata = chunks[0].get("metadata", {})
            title = metadata.get("title", "Unknown")
            document_type = metadata.get("document_type", "Unknown")

            # Tag each chunk with its page/location (when known) so the
            # model can still cite the right spot even though chunks from
            # the same document are grouped together.
            chunk_texts = []
            for chunk in chunks:
                location = prompt_lib._location_tag(chunk.get("metadata", {}))
                chunk_texts.append(f"[{location.strip(' ()') or 'excerpt'}]\n{chunk['text']}")

            document_text = "\n\n".join(chunk_texts)

            parts.append(
                f"Document Title: {title}\n"
                f"Document Type: {document_type}\n\n"
                f"{document_text}"
            )

        return f"\n\n{'-' * 60}\n\n".join(parts)
