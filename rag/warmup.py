"""
warmup.py

Optional process-start warm-up so the first user request does not pay
model-load latency for the embedding model and the cross-encoder.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def warm_models() -> None:
    """Load embedding + reranker models into the process-wide cache."""
    try:
        from .search import warm_embedding_model
        warm_embedding_model()
    except Exception as exc:
        logger.warning("Embedding warm-up failed: %s", exc)

    try:
        from .reranker import warm_reranker_model
        warm_reranker_model()
    except Exception as exc:
        logger.warning("Reranker warm-up failed: %s", exc)

    logger.info("Model warm-up complete.")
