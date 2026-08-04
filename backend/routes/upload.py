"""
upload.py

POST /upload validates the incoming files (non-empty, within
MAX_UPLOAD_SIZE_MB) and returns a job id immediately; the actual parse ->
chunk -> embed -> index pipeline runs on a worker thread via
asyncio.to_thread (the same pattern routes/ws.py already uses for
streaming), so a large file can't block the event loop for its whole
ingestion time or blow past the frontend's request timeout. The frontend
polls GET /upload/status/{job_id} for real stage progress and a
categorized error message if ingestion fails.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from config import settings
from services import upload_jobs
from services.rag_service import update_db_files, list_stored_files, get_stored_file_bytes
from services.storage_service import StorageUnavailableError

log = logging.getLogger("routes.upload")

router = APIRouter()


def _categorize_error(stage: str, exc: Exception) -> str:
    """
    Turn a raw exception into a message that tells the user roughly what
    went wrong, using the stage ingestion had reached when it failed —
    much more actionable than a bare str(e) from an arbitrary internal
    failure.
    """
    if isinstance(exc, StorageUnavailableError):
        return f"File storage is currently unavailable: {exc}"
    if stage in ("queued", "parsing"):
        return f"Could not read or parse one of the uploaded files: {exc}"
    if stage == "chunking":
        return f"Could not process the uploaded document's content: {exc}"
    if stage == "embedding":
        return f"Could not index the document for search: {exc}"
    return f"Ingestion failed: {exc}"


def _ingest_job(job_id: str, file_dicts: List[dict]) -> None:
    last_stage = "queued"

    def _on_progress(stage: str) -> None:
        nonlocal last_stage
        last_stage = stage
        upload_jobs.set_stage(job_id, stage)

    try:
        chunks_added = update_db_files(file_dicts, on_progress=_on_progress)
        upload_jobs.mark_done(job_id, chunks_added)
    except Exception as e:
        log.exception(f"Background ingestion failed for job {job_id}")
        upload_jobs.mark_error(job_id, _categorize_error(last_stage, e))


@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    file_dicts = []

    for f in files:
        data = await f.read()

        if not data:
            raise HTTPException(status_code=400, detail=f"File '{f.filename}' is empty.")

        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File '{f.filename}' is {len(data) / (1024 * 1024):.1f}MB, which exceeds "
                    f"the {settings.MAX_UPLOAD_SIZE_MB}MB upload limit."
                ),
            )

        file_dicts.append({
            "filename": f.filename or "unknown",
            "data": data,
        })

    job_id = upload_jobs.create_job([f["filename"] for f in file_dicts])

    # Fire-and-forget: the request returns immediately with a job id, well
    # before the frontend's request timeout, regardless of how long
    # ingestion actually takes.
    asyncio.create_task(asyncio.to_thread(_ingest_job, job_id, file_dicts))

    return {"job_id": job_id, "status": "queued"}


@router.get("/upload/status/{job_id}")
async def upload_status(job_id: str):
    job = upload_jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown upload job (it may have expired).")

    response = {
        "job_id": job["job_id"],
        "status": job["status"],
        "stage": job["stage"],
    }
    if job["status"] == "done":
        response["chunks_added"] = job["chunks_added"]
        response["stored_files"] = list_stored_files()
    if job["status"] == "error":
        response["error"] = job["error"]
    return response


@router.get("/stored-files")
async def stored_files():
    return {
        "files": list_stored_files()
    }


@router.get("/files/{object_name}/download")
async def download_stored_file(object_name: str):
    """Stream a previously uploaded file's bytes back from MinIO. Used as
    a fallback when the client can't reach MinIO's presigned URL directly
    (e.g. MinIO bound to an internal Docker hostname)."""
    try:
        data = get_stored_file_bytes(object_name)
    except StorageUnavailableError as e:
        raise HTTPException(status_code=503, detail=f"File storage is currently unavailable: {e}")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File not found: {e}")

    return StreamingResponse(
        iter([data]),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{object_name}"'},
    )
