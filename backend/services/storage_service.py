"""
storage_service.py

Thin wrapper around the MinIO client used as object storage for:
- originally uploaded documents (bucket: settings.MINIO_BUCKET_UPLOADS),
  replacing the old "save to local disk" behaviour in rag_service.py
- generated PDF reports (bucket: settings.MINIO_BUCKET_REPORTS)

This is the ONLY module that talks to MinIO directly; everything else
(rag_service, report_service, routes) goes through the small functions
exposed here so the rest of the app never imports the `minio` SDK.
"""

from __future__ import annotations

import io
import logging
from datetime import timedelta
from typing import Optional

from minio import Minio
from minio.error import S3Error

from config import settings

log = logging.getLogger("storage_service")

_client: Optional[Minio] = None

# Buckets are created lazily on first use and cached here so we don't hit
# MinIO's bucket-exists API on every single upload/download call.
_ensured_buckets: set[str] = set()


def get_client() -> Minio:
    global _client
    if _client is None:
        _client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        log.info("MinIO client initialised at %s (secure=%s)", settings.MINIO_ENDPOINT, settings.MINIO_SECURE)
    return _client


def ensure_bucket(bucket: str) -> None:
    if bucket in _ensured_buckets:
        return
    client = get_client()
    try:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            log.info("Created MinIO bucket '%s'", bucket)
    except S3Error as e:
        log.error("Could not ensure bucket '%s': %s", bucket, e)
        raise
    _ensured_buckets.add(bucket)


def is_available() -> bool:
    """Best-effort connectivity check, used by the health endpoint."""
    try:
        get_client().list_buckets()
        return True
    except Exception as e:
        log.warning("MinIO not reachable: %s", e)
        return False


def upload_bytes(
    bucket: str,
    object_name: str,
    data: bytes,
    content_type: str = "application/octet-stream",
) -> str:
    """Upload raw bytes to `bucket/object_name`. Returns the object name."""
    ensure_bucket(bucket)
    client = get_client()
    client.put_object(
        bucket,
        object_name,
        data=io.BytesIO(data),
        length=len(data),
        content_type=content_type,
    )
    return object_name


def download_bytes(bucket: str, object_name: str) -> bytes:
    """Download and return the full contents of `bucket/object_name`."""
    client = get_client()
    response = client.get_object(bucket, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def object_exists(bucket: str, object_name: str) -> bool:
    client = get_client()
    try:
        client.stat_object(bucket, object_name)
        return True
    except S3Error:
        return False


def delete_object(bucket: str, object_name: str) -> None:
    client = get_client()
    try:
        client.remove_object(bucket, object_name)
    except S3Error as e:
        log.warning("Could not delete '%s/%s': %s", bucket, object_name, e)


def presigned_url(bucket: str, object_name: str, expiry_seconds: Optional[int] = None) -> Optional[str]:
    """Return a temporary, directly-downloadable URL for an object, or
    None if MinIO isn't reachable (the caller can fall back to proxying
    the download through the backend instead)."""
    client = get_client()
    try:
        return client.presigned_get_object(
            bucket,
            object_name,
            expires=timedelta(seconds=expiry_seconds or settings.MINIO_PRESIGNED_EXPIRY),
        )
    except Exception as e:
        log.warning("Could not build presigned URL for '%s/%s': %s", bucket, object_name, e)
        return None
