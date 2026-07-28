"""
reranker.py

Re-rank retrieved chunks using a CrossEncoder (BGE by default).

Latency notes
-------------
- Model is loaded once per process and shared.
- Device (cuda/cpu) configurable via RAG_DEVICE.
- Model name configurable via RAG_RERANKER_MODEL (can point at a lighter
  cross-encoder if quality permits).
- Default threshold raised to 0.4 (was 0.3) to drop marginal candidates
  earlier; override with RAG_RERANK_THRESHOLD.
"""

from __future__ import annotations

import logging
import os
import threading
import time

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_model = None
_init_lock = threading.Lock()

_LOG_LATENCY = os.getenv("RAG_LOG_LATENCY", "false").strip().lower() in ("3", "true", "yes")


def _resolve_device() -> str:
    device = os.getenv("RAG_DEVICE", "auto").strip().lower()
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device if device in ("cuda", "cpu") else "cpu"


def _get_reranker_model() -> CrossEncoder:
    global _model
    if _model is None:
        with _init_lock:
            if _model is None:
                model_name = os.getenv(
                    "RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
                )
                device = _resolve_device()
                t0 = time.perf_counter()
                # device is passed via model_kwargs / to() depending on version;
                # CrossEncoder accepts device= in recent sentence-transformers.
                try:
                    _model = CrossEncoder(model_name, device=device)
                except TypeError:
                    _model = CrossEncoder(model_name)
                    if device == "cuda":
                        try:
                            _model.model.to(device)
                        except Exception:
                            pass
                if _LOG_LATENCY:
                    logger.info(
                        "[latency] reranker model load (%.2fs) model=%s device=%s",
                        time.perf_counter() - t0,
                        model_name,
                        device,
                    )
    return _model


def warm_reranker_model() -> None:
    """Force-load the reranker at process start (optional warm-up)."""
    _get_reranker_model()


class Reranker:

    def __init__(self):
        self.model = _get_reranker_model()
        self.default_threshold = float(os.getenv("RAG_RERANK_THRESHOLD", "0.4"))

    def rank(
        self,
        question: str,
        documents: list[dict],
        top_k: int = 5,
        threshold: float | None = None,
    ):
        if not documents:
            return []

        threshold = self.default_threshold if threshold is None else threshold

        t0 = time.perf_counter()

        pairs = [(question, document["text"]) for document in documents]
        scores = self.model.predict(pairs)

        for document, score in zip(documents, scores):
            document["rerank_score"] = float(score)

        documents.sort(key=lambda x: x["rerank_score"], reverse=True)

        filtered = [
            doc for doc in documents if doc["rerank_score"] >= threshold
        ]

        if _LOG_LATENCY:
            logger.info(
                "[latency] rerank candidates=%s kept=%s total=%.3fs threshold=%.2f",
                len(documents),
                len(filtered),
                time.perf_counter() - t0,
                threshold,
            )

        return filtered[:top_k]
