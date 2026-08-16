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
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

from config import settings
from services import upload_jobs
from services.rag_service import (
    delete_document,
    find_registry_entry_by_document_id,
    get_stored_file_bytes,
    list_stored_files,
    reindex_document,
    update_db_files,
)
from services.storage_service import StorageUnavailableError
from utils.file_validation import InvalidFileError, UnsupportedFileTypeError, validate_upload

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


def _ingest_job(job_id: str, file_dicts: List[dict], conversation_id: str) -> None:
    last_stage = "queued"

    def _on_progress(stage: str) -> None:
        nonlocal last_stage
        last_stage = stage
        upload_jobs.set_stage(job_id, stage)

    try:
        chunks_added = update_db_files(
            file_dicts, conversation_id=conversation_id, on_progress=_on_progress
        )
        upload_jobs.mark_done(job_id, chunks_added)
    except Exception as e:
        log.exception(f"Background ingestion failed for job {job_id}")
        upload_jobs.mark_error(job_id, _categorize_error(last_stage, e))


@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    # No default: which conversation an upload belongs to must always be
    # explicit — a silent fallback here would tag documents with the wrong
    # (or a shared) conversation_id and defeat Document Isolation just as
    # surely as the old conversation_id fallback did. See
    # frontend/lib/conversation.ts for the per-tab id this must carry.
    conversation_id: str = Form(...),
):
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

        filename = f.filename or "unknown"
        try:
            validate_upload(filename, data)
        except UnsupportedFileTypeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except InvalidFileError as e:
            raise HTTPException(status_code=400, detail=str(e))

        file_dicts.append({
            "filename": filename,
            "data": data,
        })

    job_id = upload_jobs.create_job([f["filename"] for f in file_dicts])

    # Fire-and-forget: the request returns immediately with a job id, well
    # before the frontend's request timeout, regardless of how long
    # ingestion actually takes.
    asyncio.create_task(asyncio.to_thread(_ingest_job, job_id, file_dicts, conversation_id))

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


# ── Document lifecycle: re-index / delete (Tasks 3 & 4) ─────────────────────

def _categorize_reindex_error(stage: str, exc: Exception) -> str:
    if isinstance(exc, ValueError):
        return str(exc)
    if isinstance(exc, RuntimeError):
        return str(exc)
    if isinstance(exc, StorageUnavailableError):
        return f"File storage is currently unavailable: {exc}"
    return _categorize_error(stage, exc)


def _reindex_job(job_id: str, document_id: str, conversation_id: str, new_file: Optional[dict]) -> None:
    last_stage = "queued"

    def _on_progress(stage: str) -> None:
        nonlocal last_stage
        last_stage = stage
        upload_jobs.set_stage(job_id, stage)

    try:
        new_filename = new_file["filename"] if new_file else None
        new_data = new_file["data"] if new_file else None
        chunks_added = reindex_document(
            document_id,
            conversation_id,
            new_filename=new_filename,
            new_data=new_data,
            on_progress=_on_progress,
        )
        upload_jobs.mark_done(job_id, chunks_added)
    except Exception as e:
        log.exception(f"Background re-index failed for job {job_id} (document_id={document_id!r})")
        upload_jobs.mark_error(job_id, _categorize_reindex_error(last_stage, e))


@router.post("/documents/{document_id}/reindex")
async def reindex_document_route(
    document_id: str,
    conversation_id: str = Form(...),
    file: Optional[UploadFile] = File(None),
):
    """
    Re-index an already-ingested document — either reprocessing its
    original stored bytes (no `file`, e.g. after a chunking/embedding/OCR
    fix or config change) or replacing its content (`file` given). Returns
    a job id polled the same way as POST /upload (GET /upload/status/{job_id})
    since re-indexing a large document goes through the same
    parse -> chunk -> embed pipeline and shouldn't block the request.

    Old vectors for this document_id are guaranteed removed before the
    job reports success — see services.rag_service.reindex_document.
    """
    if find_registry_entry_by_document_id(document_id, conversation_id) is None:
        raise HTTPException(
            status_code=404,
            detail=f"No document found with id '{document_id}' for this conversation.",
        )

    new_file: Optional[dict] = None
    if file is not None:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' is empty.")
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File '{file.filename}' is {len(data) / (1024 * 1024):.1f}MB, which exceeds "
                    f"the {settings.MAX_UPLOAD_SIZE_MB}MB upload limit."
                ),
            )
        new_filename = file.filename or "unknown"
        try:
            validate_upload(new_filename, data)
        except UnsupportedFileTypeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except InvalidFileError as e:
            raise HTTPException(status_code=400, detail=str(e))
        new_file = {"filename": new_filename, "data": data}

    job_id = upload_jobs.create_job([new_file["filename"]] if new_file else [])

    asyncio.create_task(
        asyncio.to_thread(_reindex_job, job_id, document_id, conversation_id, new_file)
    )

    return {"job_id": job_id, "status": "queued"}


@router.delete("/documents/{document_id}")
async def delete_document_route(document_id: str, conversation_id: str):
    """
    Delete a document and every one of its indexed chunks from Qdrant.
    Idempotent — deleting an already-deleted (or never-existing)
    document_id returns chunks_removed=0 rather than an error.
    """
    try:
        removed = await asyncio.to_thread(delete_document, document_id, conversation_id)
    except Exception as e:
        log.exception(f"Failed to delete document_id={document_id!r}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e}")

    return {"document_id": document_id, "chunks_removed": removed}
