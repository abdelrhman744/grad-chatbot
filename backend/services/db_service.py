"""
db_service.py

Thin wrapper around the Qdrant client used as the vector store for both
the RAG ingestion pipeline and the agent's retrieval tool.

Qdrant runs as a separate server (the `qdrant` service in
docker-compose.yml, or a locally-run Qdrant server for native dev) — this
module talks to it over HTTP via QDRANT_URL, never via embedded/file mode.
Every network call is wrapped in `_with_retries` so a container that's
still starting, or a brief network blip, doesn't immediately fail the
caller (see requirement: graceful degradation instead of hard failure).
"""

import logging
import time
from typing import Callable, TypeVar

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from config import settings

log = logging.getLogger("db_service")

_client: QdrantClient | None = None

T = TypeVar("T")


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
            timeout=settings.QDRANT_TIMEOUT_SECONDS,
        )
        log.info("Qdrant client initialised for %s", settings.QDRANT_URL)
    return _client


def get_collection_name() -> str:
    return settings.QDRANT_COLLECTION


def with_retries(
    fn: Callable[[], T],
    *,
    what: str,
    retries: int | None = None,
    delay: float | None = None,
) -> T:
    """Run `fn`, retrying on any exception up to `retries` times with a
    fixed delay between attempts. Re-raises the last exception once
    attempts are exhausted — callers decide whether that's fatal."""
    retries = settings.QDRANT_CONNECT_RETRIES if retries is None else retries
    delay = settings.QDRANT_RETRY_DELAY_SECONDS if delay is None else delay
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - deliberately broad: any Qdrant/network failure is retryable here
            last_exc = e
            if attempt == retries:
                break
            log.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                what, attempt, retries, e, delay,
            )
            time.sleep(delay)
    log.error("%s failed after %d attempt(s): %s", what, retries, last_exc)
    raise last_exc  # type: ignore[misc]


def collection_exists(client: QdrantClient) -> bool:
    try:
        client.get_collection(settings.QDRANT_COLLECTION)
        return True
    except Exception:
        return False


def is_available() -> bool:
    """Best-effort connectivity check, used by the health endpoint. No
    retries — this is a live probe, not a startup/critical-path call."""
    try:
        get_client().get_collections()
        return True
    except Exception as e:
        log.warning("Qdrant not reachable: %s", e)
        return False


def _check_existing_schema(client: QdrantClient, expected_size: int) -> None:
    """Verify an already-existing collection's vector size/distance match
    what this app expects. Raises a clear error on mismatch instead of
    silently recreating or deleting the collection — an incompatible
    schema almost always means QDRANT_COLLECTION points at data from a
    different embedding model/config and needs a human decision, not an
    automatic fix."""
    info = client.get_collection(settings.QDRANT_COLLECTION)
    vectors_config = info.config.params.vectors
    if not isinstance(vectors_config, VectorParams):
        # Named-vector / multi-vector collection — not the shape this app
        # creates, so the simple size/distance check below doesn't apply.
        log.warning(
            "Collection '%s' uses a named-vector config this app doesn't "
            "create — skipping schema verification.",
            settings.QDRANT_COLLECTION,
        )
        return
    if vectors_config.size != expected_size or vectors_config.distance != Distance.COSINE:
        raise RuntimeError(
            f"Qdrant collection '{settings.QDRANT_COLLECTION}' already exists "
            f"with an incompatible schema (size={vectors_config.size}, "
            f"distance={vectors_config.distance}) but this app expects "
            f"(size={expected_size}, distance={Distance.COSINE}). Refusing to "
            "modify it automatically. Point QDRANT_COLLECTION at a new name, "
            "or delete the existing collection manually if this mismatch is "
            "expected (e.g. after changing EMBEDDING_MODEL)."
        )


def ensure_collection(embeddings) -> None:
    """Create the Qdrant collection if it doesn't already exist; if it
    does, verify its schema matches instead of assuming it's compatible."""

    def _do() -> None:
        client = get_client()
        sample_vec = embeddings.embed_query("test")
        vector_size = len(sample_vec)
        if collection_exists(client):
            _check_existing_schema(client, vector_size)
            return
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        log.info(
            "Collection '%s' created — vector size %d",
            settings.QDRANT_COLLECTION, vector_size,
        )

    with_retries(_do, what="ensure_collection")
