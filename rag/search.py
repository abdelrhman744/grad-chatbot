"""
search.py

Dense vector search using Qdrant.

Latency notes
-------------
- Embedding model + Qdrant client are process-shared (loaded once).
- Query embeddings are LRU-cached (normalized query string) to avoid
  re-encoding identical / near-identical questions.
- Device (cuda/cpu) is configurable via RAG_DEVICE.
- Embedding model name is configurable via RAG_EMBEDDING_MODEL.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

_model = None
_client = None
_init_lock = threading.Lock()

# Simple process-wide LRU for query embeddings (avoids repeated encodes
# of the same question within a process lifetime).
_EMBED_CACHE_MAX = int(os.getenv("RAG_EMBED_CACHE_SIZE", "256"))
_embed_cache: OrderedDict[str, list[float]] = OrderedDict()
_embed_cache_lock = threading.Lock()

_LOG_LATENCY = os.getenv("RAG_LOG_LATENCY", "false").strip().lower() in ("1", "true", "yes")


def _resolve_device() -> str:
    device = os.getenv("RAG_DEVICE", "auto").strip().lower()
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device if device in ("cuda", "cpu") else "cpu"


def _get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _init_lock:
            if _model is None:
                t0 = time.perf_counter()
                # Prefer the shared vector_db singleton so indexing + search
                # use the exact same loaded model (one process-wide load).
                try:
                    from vector_db.model_manager import get_embedding_model as _shared
                    _model = _shared()
                except Exception:
                    model_name = os.getenv(
                        "RAG_EMBEDDING_MODEL", "intfloat/multilingual-e5-large"
                    )
                    device = _resolve_device()
                    _model = SentenceTransformer(model_name, device=device)
                if _LOG_LATENCY:
                    logger.info(
                        "[latency] embedding model load (%.2fs)",
                        time.perf_counter() - t0,
                    )
    return _model


def _get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        with _init_lock:
            if _client is None:
                db_path = Path(__file__).resolve().parent.parent / "vector_db" / "qdrant_db"
                _client = QdrantClient(path=str(db_path))
    return _client


def _cache_key(query: str) -> str:
    # Stable short key; full string is fine but hashing keeps the dict lean.
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _encode_query(model: SentenceTransformer, query: str) -> list[float]:
    """Encode with LRU cache. Key is the prefixed query string."""
    key = _cache_key(query)
    with _embed_cache_lock:
        if key in _embed_cache:
            _embed_cache.move_to_end(key)
            return _embed_cache[key]

    vector = model.encode(query, normalize_embeddings=True).tolist()

    with _embed_cache_lock:
        _embed_cache[key] = vector
        if len(_embed_cache) > _EMBED_CACHE_MAX:
            _embed_cache.popitem(last=False)
    return vector


def warm_embedding_model() -> None:
    """Force-load the embedding model at process start (optional warm-up)."""
    _get_embedding_model()


class SearchEngine:

    def __init__(self):
        self.model = _get_embedding_model()
        self.client = _get_qdrant_client()
        self.collection_name = "documents"

    def search(self, query: str, top_k: int = 5):
        t0 = time.perf_counter()

        # E5 models expect the "query: " prefix
        prefixed = f"query: {query}"
        query_vector = _encode_query(self.model, prefixed)

        t_embed = time.perf_counter()

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        if _LOG_LATENCY:
            logger.info(
                "[latency] dense_search embed=%.3fs search=%.3fs total=%.3fs top_k=%s",
                t_embed - t0,
                time.perf_counter() - t_embed,
                time.perf_counter() - t0,
                top_k,
            )

        return results.points
