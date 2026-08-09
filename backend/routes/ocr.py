"""
routes/ocr.py

POST /api/ocr/handwritten — free, local handwritten OCR (Arabic + English)
via services/handwritten_ocr_service.py (Hugging Face TrOCR). Separate from
the printed-text OCR already embedded in the document upload pipeline
(services/ocr_service.py, Tesseract, routes/upload.py) — this route is for
a user explicitly running OCR on one handwritten image (a photo/scan of a
note, a cropped text line, etc.) and getting the extracted text back
directly, rather than indexing a whole document.

TrOCR handwritten checkpoints are trained on single text-line images, not
full multi-paragraph pages. A full-page image no longer needs to be
pre-cropped by the caller: services/handwritten_ocr_service.py
automatically splits it into individual line crops (classic OpenCV
ink-density segmentation, no extra ML model), OCRs each line, and joins
the results — see HANDWRITTEN_OCR.md for how that works and its
limitations (it assumes roughly horizontal, non-overlapping lines).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from config import settings
from services.handwritten_ocr_service import (
    SUPPORTED_LANGUAGES,
    InvalidImageError,
    OCRModelError,
    UnsupportedLanguageError,
    get_handwritten_ocr_service,
)
from services.rag_service import update_db_files

log = logging.getLogger("routes.ocr")

router = APIRouter()

_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}


def _ext(filename: str) -> str:
    return filename.lower().rsplit(".", 1)[-1] if "." in filename else ""


def _run_ocr(data: bytes, language: str) -> str:
    return get_handwritten_ocr_service().recognize(data, language)


@router.post("/ocr/handwritten")
async def ocr_handwritten(
    file: UploadFile = File(...),
    language: str = Form(...),
    # Optional integration with the existing document pipeline (see
    # docs/handwritten-ocr.md): when BOTH are provided, the extracted text
    # is fed into the exact same chunk -> embed -> Qdrant pipeline every
    # other upload goes through (as a synthetic .txt "file"), tagged with
    # this conversation_id exactly like any other upload — see
    # services/rag_service.update_db_files. Off by default so a plain
    # "just OCR this image" call stays a single fast, side-effect-free
    # request, matching the response shape in the feature spec.
    conversation_id: Optional[str] = Form(None),
    index: bool = Form(False),
):
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported language {language!r}. Handwritten OCR only supports "
                f"explicit {list(SUPPORTED_LANGUAGES)} — automatic language detection "
                "isn't offered here because it can't be done reliably before OCR has "
                "actually run (see docs/handwritten-ocr.md)."
            ),
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    if _ext(file.filename) not in _IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '.{_ext(file.filename)}'. "
                f"Accepted image types: {sorted(_IMAGE_EXTENSIONS)}"
            ),
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File is {len(data) / (1024 * 1024):.1f}MB, which exceeds "
                f"the {settings.MAX_UPLOAD_SIZE_MB}MB upload limit."
            ),
        )

    try:
        # Model load (first call per language) + inference are blocking
        # CPU/GPU work — off the event loop, same pattern used for the
        # agent's blocking LLM calls in routes/chat.py.
        text = await asyncio.to_thread(_run_ocr, data, language)
    except UnsupportedLanguageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    except OCRModelError as e:
        log.exception("Handwritten OCR model error")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        log.exception("Unexpected handwritten OCR error")
        raise HTTPException(status_code=500, detail=f"Handwritten OCR failed: {e}")

    response: dict = {"text": text, "language": language, "type": "handwritten"}

    if index and conversation_id:
        if not text.strip():
            response["indexed"] = False
        else:
            stem = file.filename.rsplit(".", 1)[0]
            try:
                chunks_added = await asyncio.to_thread(
                    update_db_files,
                    [{"filename": f"{stem}.handwritten-ocr.txt", "data": text.encode("utf-8")}],
                    conversation_id,
                )
                response["indexed"] = True
                response["chunks_added"] = chunks_added
            except Exception as e:
                log.exception("Failed to index handwritten OCR text into the RAG pipeline")
                response["indexed"] = False
                response["index_error"] = str(e)

    return response
