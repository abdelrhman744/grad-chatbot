"""
embeddings_provider.py

Single, reusable local embedding service. Embeddings are produced entirely
on-device with `sentence-transformers` — no embedding API, no API key, no
network call at inference time (only the one-time model download on first
run, cached by HuggingFace's local cache afterwards).

Design:
- `SentenceTransformer(settings.EMBEDDING_MODEL)` is loaded exactly ONCE,
  the first time `get_embeddings()` is called (typically triggered at
  application startup by `rag_service.py`'s module-level
  `embeddings = get_embeddings()`), and kept as a module-level singleton.
- The SAME model instance is reused for both document indexing
  (`embed_documents`) and query embedding (`embed_query`) — it is never
  reloaded per request.
- `LocalEmbeddings` exposes `embed_documents(list[str]) -> list[list[float]]`
  and `embed_query(str) -> list[float]`, the exact duck-typed interface
  `langchain_qdrant.QdrantVectorStore` (and `services/db_service.py`,
  which only calls `.embed_query(...)`) already expect — so Qdrant itself
  requires zero changes.
- Vectors are L2-normalized (`normalize_embeddings=True`) before being
  returned/stored, as required for cosine-similarity search in Qdrant
  (the collection is created with `Distance.COSINE` in `db_service.py`).

intfloat/multilingual-e5-large is an "E5" model: it was trained with
"query: " / "passage: " instructional prefixes, and using them measurably
improves retrieval quality. They're applied here transparently (the rest
of the codebase just calls `embed_query` / `embed_documents` with plain
text, unaware of the prefixing) — this does not change the public
interface, configuration, or Qdrant compatibility in any way.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.embeddings import Embeddings

from config import settings
from utils import timing
from utils.device import resolve_device

log = logging.getLogger("embeddings_provider")


class LocalEmbeddings(Embeddings):
    """
    Loads a `sentence-transformers` model once and reuses it for every
    embedding call (document indexing and query embedding alike).
    """

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        device = resolve_device()
        log.info(f"Loading local embedding model '{model_name}' on device={device!r} (one-time load)...")
        self._model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        log.info(f"Embedding model '{model_name}' loaded and ready on {device!r}.")

    def _encode(self, texts: List[str]) -> List[List[float]]:
        # Cumulative compute time across every embed call in the request
        # (query embeddings for each retrieval variant, document embeddings
        # for MMR diversification, ...). Recorded as a substage — see
        # utils/timing.py — since these calls run concurrently across
        # threads inside a wider "embedding_and_qdrant_retrieval" stage, so
        # summing their durations is CPU-time, not wall-time.
        with timing.substage("embedding_model_compute"):
            vectors = self._model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        return vectors.tolist()

    # ── langchain-compatible Embeddings interface ──────────────────────────
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document chunks for indexing into Qdrant."""
        prefixed = [f"passage: {t}" for t in texts]
        return self._encode(prefixed)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single user query for similarity search against Qdrant."""
        return self._encode([f"query: {text}"])[0]

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        """
        Batched counterpart of embed_query: embed several query strings
        (e.g. every retrieval query variant) in ONE forward pass instead of
        one call each. Firing N separate embed_query() calls concurrently
        across threads (the previous retrieval pattern) is fine on CPU
        (independent cores) but actively hurts on GPU — a single device
        processes concurrently-submitted small calls with far more
        per-call overhead than one batched call of the same total size
        (measured ~10x slower for 9 single-item concurrent GPU calls vs.
        one batched call of the same 9 texts — see PROFILING.md). Batching
        here is a straight win on both CPU and GPU.
        """
        if not texts:
            return []
        return self._encode([f"query: {t}" for t in texts])


# ── Shared singleton ─────────────────────────────────────────────────────────
_embeddings: Optional[LocalEmbeddings] = None


def get_embeddings() -> LocalEmbeddings:
    """
    Return the shared embedding model instance, loading it on first call
    only. Every subsequent call (indexing or querying, from any request)
    reuses this same in-memory model — it is never reloaded.
    """
    global _embeddings
    if _embeddings is None:
        provider = (settings.EMBEDDING_PROVIDER or "local").lower()
        if provider != "local":
            log.warning(
                f"EMBEDDING_PROVIDER='{provider}' is not supported — this service "
                "only runs local sentence-transformers models. Falling back to 'local'."
            )
        _embeddings = LocalEmbeddings(settings.EMBEDDING_MODEL)
    return _embeddings
