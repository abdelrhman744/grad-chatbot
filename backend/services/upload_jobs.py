"""
upload_jobs.py

In-process registry of background upload/ingestion jobs. POST /upload
returns a job id immediately instead of blocking on the full parse ->
chunk -> embed -> index pipeline (which can take anywhere from seconds to
minutes depending on file size/type); the frontend then polls
GET /upload/status/{job_id} for real stage progress instead of staring at
an opaque spinner for the whole ingestion duration.

This is an in-memory, single-process registry — the same tradeoff already
made by the per-conversation Agent registry (agent/session.py) and the
Qdrant client singleton (services/rag_service.py): jobs are lost on
backend restart, which is acceptable given the app already assumes one
long-lived process.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Optional

_jobs: dict[str, dict] = {}
_lock = threading.Lock()

# Finished jobs older than this are pruned on the next lookup so a
# long-running process doesn't accumulate them forever.
_JOB_TTL_SECONDS = 3600


def create_job(filenames: list[str]) -> str:
    job_id = uuid.uuid4().hex
    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",  # queued | processing | done | error
            "stage": "queued",  # queued | parsing | chunking | embedding | done
            "filenames": filenames,
            "chunks_added": None,
            "error": None,
            "created_at": time.time(),
        }
    return job_id


def set_stage(job_id: str, stage: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            job["stage"] = stage
            job["status"] = "processing"


def mark_done(job_id: str, chunks_added: int) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            job["status"] = "done"
            job["stage"] = "done"
            job["chunks_added"] = chunks_added


def mark_error(job_id: str, message: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            job["status"] = "error"
            job["error"] = message


def get_job(job_id: str) -> Optional[dict]:
    _prune_expired()
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job is not None else None


def _prune_expired() -> None:
    cutoff = time.time() - _JOB_TTL_SECONDS
    with _lock:
        expired = [jid for jid, j in _jobs.items() if j["created_at"] < cutoff]
        for jid in expired:
            del _jobs[jid]
