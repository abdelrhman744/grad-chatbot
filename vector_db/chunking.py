"""
chunking.py

Semantic text chunking using multilingual-e5-large.

multilingual-e5-large has a max sequence length of 512 tokens.
We keep chunks under that limit (including the "passage: " prefix)
and never call the tokenizer in a way that triggers the
"sequence length is longer than ..." warning.
"""

from __future__ import annotations

import re

from sklearn.metrics.pairwise import cosine_similarity

from .model_manager import get_embedding_model, get_tokenizer

# E5 max length is 512; reserve room for the "passage: " prefix (~2 tokens).
_MODEL_MAX_TOKENS = 512
_PREFIX_RESERVE = 8
_SAFE_MAX_TOKENS = _MODEL_MAX_TOKENS - _PREFIX_RESERVE  # 504


class SemanticChunker:

    def __init__(
        self,
        similarity_threshold=0.82,
        max_tokens=450,
        max_sentences=8,
        overlap_sentences=1,
    ):
        self.model = get_embedding_model()
        self.tokenizer = get_tokenizer()

        # Cap at the safe limit for this embedding model.
        self.max_tokens = min(max_tokens, _SAFE_MAX_TOKENS)
        self.similarity_threshold = similarity_threshold
        self.max_sentences = max_sentences
        self.overlap_sentences = overlap_sentences

        # Ensure SentenceTransformer truncates at encode-time instead of erroring.
        try:
            self.model.max_seq_length = _MODEL_MAX_TOKENS
        except Exception:
            pass

    # =====================================================
    # Public Method
    # =====================================================

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []

        paragraphs = self._split_paragraphs(text)
        final_chunks: list[str] = []

        for paragraph in paragraphs:
            sentences = self._split_sentences(paragraph)
            if not sentences:
                continue

            # Break any single sentence that is already over the token budget.
            expanded: list[str] = []
            for sentence in sentences:
                expanded.extend(self._force_split(sentence))
            sentences = expanded
            if not sentences:
                continue

            if len(sentences) == 1:
                final_chunks.append(sentences[0])
                continue

            embeddings = self._embed(sentences)
            paragraph_chunks = self._merge(sentences, embeddings)
            paragraph_chunks = self._add_overlap(paragraph_chunks)
            final_chunks.extend(paragraph_chunks)

        # Final safety: never emit a chunk over the model limit.
        safe: list[str] = []
        for c in final_chunks:
            safe.extend(self._force_split(c))
        return safe

    # =====================================================
    # Paragraph / sentence split
    # =====================================================

    def _split_paragraphs(self, text: str) -> list[str]:
        paragraphs = re.split(r"\n{2,}", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_sentences(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!؟])\s+|\n", text)
        return [s.strip() for s in sentences if s.strip()]

    # =====================================================
    # Embeddings
    # =====================================================

    def _embed(self, sentences: list[str]):
        # Truncate each sentence for embedding so encode never sees >512 tokens.
        passages = []
        for sentence in sentences:
            clipped = self._truncate_to_tokens(sentence, _SAFE_MAX_TOKENS)
            passages.append("passage: " + clipped)

        return self.model.encode(
            passages,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    # =====================================================
    # Merge similar sentences
    # =====================================================

    def _merge(self, sentences, embeddings):
        chunks: list[list[str]] = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            similarity = cosine_similarity(
                embeddings[i - 1].reshape(1, -1),
                embeddings[i].reshape(1, -1),
            )[0][0]

            candidate = " ".join(current_chunk + [sentences[i]])
            token_count = self._count_tokens(candidate)

            if (
                similarity >= self.similarity_threshold
                and token_count <= self.max_tokens
                and len(current_chunk) < self.max_sentences
            ):
                current_chunk.append(sentences[i])
            else:
                chunks.append(current_chunk)
                current_chunk = [sentences[i]]

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    # =====================================================
    # Sentence overlap
    # =====================================================

    def _add_overlap(self, chunks: list[list[str]]) -> list[str]:
        if self.overlap_sentences == 0:
            return [" ".join(chunk) for chunk in chunks]

        overlapped: list[str] = []
        previous: list[str] = []

        for chunk in chunks:
            merged = previous + chunk
            text = " ".join(merged)
            # Drop overlap if it would push the chunk over the limit.
            if self._count_tokens(text) > self.max_tokens:
                text = " ".join(chunk)
            overlapped.append(text)
            previous = chunk[-self.overlap_sentences :]

        return overlapped

    # =====================================================
    # Token helpers (no transformers length warning)
    # =====================================================

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens without triggering transformers' "sequence longer than
        model_max_length" warning (we only use this for split decisions).
        """
        prev = getattr(self.tokenizer, "model_max_length", _MODEL_MAX_TOKENS)
        try:
            self.tokenizer.model_max_length = int(1e9)
            return len(
                self.tokenizer.encode(text, add_special_tokens=False)
            )
        finally:
            self.tokenizer.model_max_length = prev

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        prev = getattr(self.tokenizer, "model_max_length", _MODEL_MAX_TOKENS)
        try:
            self.tokenizer.model_max_length = int(1e9)
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) <= max_tokens:
                return text
            ids = ids[:max_tokens]
            return self.tokenizer.decode(ids, skip_special_tokens=True)
        finally:
            self.tokenizer.model_max_length = prev

    def _force_split(self, text: str) -> list[str]:
        """Split text into pieces each <= max_tokens."""
        if self._count_tokens(text) <= self.max_tokens:
            return [text] if text.strip() else []

        # Prefer splitting on spaces so we don't break mid-word.
        words = text.split()
        pieces: list[str] = []
        current: list[str] = []

        for word in words:
            trial = (" ".join(current + [word])).strip()
            if current and self._count_tokens(trial) > self.max_tokens:
                pieces.append(" ".join(current))
                current = [word]
            else:
                current.append(word)

        if current:
            pieces.append(" ".join(current))

        # Extremely long single "words" — hard truncate by tokens.
        out: list[str] = []
        for p in pieces:
            if self._count_tokens(p) <= self.max_tokens:
                out.append(p)
            else:
                out.append(self._truncate_to_tokens(p, self.max_tokens))
        return [p for p in out if p.strip()]