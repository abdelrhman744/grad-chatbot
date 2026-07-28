"""
embedding_service.py

Thin wrapper around the shared embedding model for passage encoding
during indexing. Uses the same process-wide singleton as model_manager /
rag.search so the model is never reloaded per upload.
"""

from __future__ import annotations

from .model_manager import get_embedding_model


class EmbeddingService:
    """Static helpers for batch passage embedding (E5 'passage: ' prefix)."""

    @staticmethod
    def embed_passages(texts: list[str]):
        if not texts:
            return []

        model = get_embedding_model()
        prefixed = [f"passage: {t}" for t in texts]
        return model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
