from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List

from services.rag_service import update_db_files, list_stored_files

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