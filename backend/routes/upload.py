from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from typing import List

from services.rag_service import update_db_files, list_stored_files, get_stored_file_bytes

router = APIRouter()


@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    file_dicts = []

    for f in files:
        data = await f.read()

        if not data:
            raise HTTPException(
                status_code=400,
                detail=f"File '{f.filename}' is empty."
            )

        file_dicts.append({
            "filename": f.filename or "unknown",
            "data": data,
        })

    try:
        chunks_added = update_db_files(file_dicts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return {
        "message": f"Processed {len(files)} file(s) successfully.",
        "chunks_added": chunks_added,
        "stored_files": list_stored_files(),
    }


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
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File not found: {e}")

    return StreamingResponse(
        iter([data]),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{object_name}"'},
    )