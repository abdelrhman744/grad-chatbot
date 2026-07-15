"""
db_service.py

Thin wrapper around the Qdrant client used as the vector store for both
the RAG ingestion pipeline and the agent's retrieval tool.
"""

import logging

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from config import settings

log = logging.getLogger("db_service")

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=settings.QDRANT_PATH)
        log.info("Qdrant client initialised at %s", settings.QDRANT_PATH)
    return _client


def get_collection_name() -> str:
    return settings.QDRANT_COLLECTION


def collection_exists(client: QdrantClient) -> bool:
    try:
        client.get_collection(settings.QDRANT_COLLECTION)
        return True
    except Exception:
        return False


def ensure_collection(embeddings) -> None:
    """Create the Qdrant collection if it doesn't already exist."""
    client = get_client()
    if collection_exists(client):
        return
    sample_vec = embeddings.embed_query("test")
    vector_size = len(sample_vec)
    client.create_collection(
        collection_name=settings.QDRANT_COLLECTION,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    log.info("Collection '%s' created — vector size %d", settings.QDRANT_COLLECTION, vector_size)
