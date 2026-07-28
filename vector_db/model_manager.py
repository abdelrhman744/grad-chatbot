"""
model_manager.py

Process-wide singleton for the embedding model and its tokenizer.
Loaded once on first use (or via warm-up at server startup) and reused
by SemanticChunker, EmbeddingService, and (via rag.search) query encoding.
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None
_lock = threading.Lock()

DEFAULT_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "intfloat/multilingual-e5-large")


def _resolve_device() -> str:
    device = os.getenv("RAG_DEVICE", "auto").strip().lower()
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device if device in ("cuda", "cpu") else "cpu"


def get_embedding_model():
    """Return the shared SentenceTransformer instance (load once)."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                device = _resolve_device()
                logger.info(
                    "Loading embedding model '%s' on %s (one-time load)...",
                    DEFAULT_MODEL,
                    device,
                )
                _model = SentenceTransformer(DEFAULT_MODEL, device=device)
                logger.info("Embedding model ready.")
    return _model


def get_tokenizer():
    """Return the shared AutoTokenizer for the embedding model (load once)."""
    global _tokenizer
    if _tokenizer is None:
        with _lock:
            if _tokenizer is None:
                from transformers import AutoTokenizer

                logger.info("Loading tokenizer for '%s' (one-time load)...", DEFAULT_MODEL)
                _tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
                logger.info("Tokenizer ready.")
    return _tokenizer


def warm_models() -> None:
    """Force-load model + tokenizer at process start."""
    get_embedding_model()
    get_tokenizer()
