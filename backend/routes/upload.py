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

Uploaded content is streamed to a temp file in bounded chunks (see
_stream_upload_to_temp_file) instead of a single unbounded `await
f.read()` — the size limit is enforced incrementally, so an oversized
file is rejected mid-stream without ever being fully buffered in memory.
Only the resulting temp file PATH is carried in the request handler's own
memory afterwards; the background job re-reads each file's bytes from
disk just before handing them to update_db_files (unchanged — still needs
the full batch of {"filename","data"} dicts at once) and always deletes
the temp files afterwards, success or failure.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
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

# Chunk size for streaming an UploadFile to disk. Small enough to keep
# peak per-file memory during the upload request low and constant
# (independent of the file's actual size), large enough that a big file
# doesn't pay excessive per-chunk Python/async overhead.
_UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1MB


async def _stream_upload_to_temp_file(f: UploadFile, max_bytes: int) -> tuple[str, int]:
    """
    Stream one UploadFile to a temp file in bounded chunks, enforcing
    `max_bytes` INCREMENTALLY — the moment the running total crosses the
    limit, the read stops and the partial temp file is removed, instead of
    reading the whole file first and checking its size afterwards. At most
    one _UPLOAD_CHUNK_SIZE chunk is ever held in memory at a time,
    regardless of how large the uploaded file actually is.

    Returns (temp_path, total_bytes_written) on success. Raises
    HTTPException(413) if the limit is exceeded (temp file already cleaned
    up before raising) — the total byte count in the message is a LOWER
    bound (bytes received so far), not the file's true final size, since
    reading further just to report an exact number would defeat the whole
    point of aborting early.
    """
    fd, tmp_path = tempfile.mkstemp(prefix="upload_", suffix=".bin")
    total = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = await f.read(_UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File '{f.filename}' exceeds the {settings.MAX_UPLOAD_SIZE_MB}MB "
                            f"upload limit (already received over {total / (1024 * 1024):.1f}MB)."
                        ),
                    )
                out.write(chunk)
        return tmp_path, total
    except Exception:
        # Any failure (size-limit abort, disk error, client disconnect,
        # ...) — never leave a partial temp file behind.
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


def _cleanup_temp_files(paths: List[str]) -> None:
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.unlink(p)
        except OSError as e:
            log.warning(f"Failed to remove temp upload file {p!r}: {e}")


def _validate_temp_file(filename: str, tmp_path: str) -> None:
    """
    Re-reads the (already size-bounded — see _stream_upload_to_temp_file)
    temp file's bytes ONCE for the existing content/magic-byte validation,
    unchanged from before. This read is bounded by MAX_UPLOAD_SIZE_MB by
    construction (the file could not have been written past that limit),
    and its result is not kept around afterwards — only the temp file path
    is carried forward in the request handler's own memory.
    """
    with open(tmp_path, "rb") as fh:
        data = fh.read()
    try:
        validate_upload(filename, data)
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except InvalidFileError as e:
        raise HTTPException(status_code=400, detail=str(e))


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


def _ingest_job(job_id: str, file_entries: List[dict], conversation_id: str) -> None:
    """
    `file_entries`: [{"filename": str, "path": str}] — temp file paths
    written by the (already-completed, already-validated) upload request.
    Runs on a worker thread (asyncio.to_thread), same as before. Reads
    each file's bytes from its temp path just before handing the full
    batch to update_db_files, which is unchanged and still expects
    {"filename", "data"} dicts all at once — see its own docstring for why
    (chunks/embeddings are computed together across the whole batch).
    Temp files are always removed afterwards, whether ingestion succeeds
    or fails.
    """
    last_stage = "queued"

    def _on_progress(stage: str) -> None:
        nonlocal last_stage
        last_stage = stage
        upload_jobs.set_stage(job_id, stage)

    try:
        file_dicts = []
        for entry in file_entries:
            with open(entry["path"], "rb") as fh:
                file_dicts.append({"filename": entry["filename"], "data": fh.read()})

        chunks_added = update_db_files(
            file_dicts, conversation_id=conversation_id, on_progress=_on_progress
        )
        upload_jobs.mark_done(job_id, chunks_added)
    except Exception as e:
        log.exception(f"Background ingestion failed for job {job_id}")
        upload_jobs.mark_error(job_id, _categorize_error(last_stage, e))
    finally:
        _cleanup_temp_files([entry["path"] for entry in file_entries])


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
    file_entries: List[dict] = []  # [{"filename": str, "path": str}]
    written_paths: List[str] = []

    try:
        for f in files:
            filename = f.filename or "unknown"

            # Streams straight to disk in bounded chunks, aborting (and
            # cleaning up its own partial file) the moment the running
            # total crosses max_bytes — the file is never fully buffered
            # in memory here, unlike the old `await f.read()`.
            tmp_path, total = await _stream_upload_to_temp_file(f, max_bytes)
            written_paths.append(tmp_path)

            if total == 0:
                raise HTTPException(status_code=400, detail=f"File '{filename}' is empty.")

            # Bounded (<= max_bytes) re-read from disk for the existing
            # content/magic-byte validation — unchanged validation logic,
            # just fed from disk instead of the original in-memory buffer.
            _validate_temp_file(filename, tmp_path)

            file_entries.append({"filename": filename, "path": tmp_path})
    except Exception:
        # Any failure partway through a multi-file upload (size limit,
        # empty file, invalid content, disconnect, ...) must not leave
        # temp files behind for files that already passed earlier in this
        # same request.
        _cleanup_temp_files(written_paths)
        raise

    job_id = upload_jobs.create_job([e["filename"] for e in file_entries])

    # Fire-and-forget: the request returns immediately with a job id, well
    # before the frontend's request timeout, regardless of how long
    # ingestion actually takes. The background job owns the temp files
    # from here on (reads + deletes them) — see _ingest_job.
    asyncio.create_task(asyncio.to_thread(_ingest_job, job_id, file_entries, conversation_id))

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


def _reindex_job(
    job_id: str, document_id: str, conversation_id: str, new_file_entry: Optional[dict]
) -> None:
    """`new_file_entry`: {"filename": str, "path": str} or None — same
    temp-file handoff as _ingest_job; the temp file (if any) is always
    removed afterwards, success or failure."""
    last_stage = "queued"

    def _on_progress(stage: str) -> None:
        nonlocal last_stage
        last_stage = stage
        upload_jobs.set_stage(job_id, stage)

    try:
        new_filename = new_file_entry["filename"] if new_file_entry else None
        new_data = None
        if new_file_entry:
            with open(new_file_entry["path"], "rb") as fh:
                new_data = fh.read()

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
    finally:
        if new_file_entry:
            _cleanup_temp_files([new_file_entry["path"]])


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

    new_file_entry: Optional[dict] = None
    if file is not None:
        filename = file.filename or "unknown"
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        tmp_path: Optional[str] = None
        try:
            # _stream_upload_to_temp_file cleans up its OWN partial file
            # before raising (e.g. size-limit abort), so tmp_path only
            # needs cleanup here for failures AFTER it returns
            # successfully (empty file, invalid content).
            tmp_path, total = await _stream_upload_to_temp_file(file, max_bytes)
            if total == 0:
                raise HTTPException(status_code=400, detail=f"File '{filename}' is empty.")
            _validate_temp_file(filename, tmp_path)
        except Exception:
            if tmp_path:
                _cleanup_temp_files([tmp_path])
            raise
        new_file_entry = {"filename": filename, "path": tmp_path}

    job_id = upload_jobs.create_job([new_file_entry["filename"]] if new_file_entry else [])

    asyncio.create_task(
        asyncio.to_thread(_reindex_job, job_id, document_id, conversation_id, new_file_entry)
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
