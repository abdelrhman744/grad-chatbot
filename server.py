"""
server.py

FastAPI backend for DocAssist AI.

- Chat / stream / reset  -> project-copy ReAct agent (rag/)
- Upload PDF+JSON        -> ocr/ pipeline then vector_db ingestion
- Upload XLSX+JSON       -> vector_db excel pipeline
- Embedding model loaded once at startup (never reloaded per upload)

Run:
    uvicorn server:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag.agent.session import DEFAULT_CONVERSATION_ID, get_agent, reset_agent
from rag.warmup import warm_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("server")

BASE_DIR = Path(__file__).resolve().parent
OCR_DATA = BASE_DIR / "ocr" / "data"
XLSX_FOLDER = BASE_DIR / "vector_db" / "xlsx"

app = FastAPI(title="DocAssist AI", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    log.info("Warming embedding / reranker models (one-time load)...")
    try:
        from vector_db.model_manager import warm_models as warm_embed

        warm_embed()
    except Exception as e:
        log.warning("vector_db model warm-up failed: %s", e)
    try:
        warm_models()
    except Exception as e:
        log.warning("rag model warm-up failed: %s", e)
    log.info("Startup complete.")


def _error_detail(e: Exception) -> str:
    msg = str(e).strip()
    return f"{type(e).__name__}: {msg}" if msg else type(e).__name__


def _safe_stem(name: str) -> str:
    return Path(name).stem.strip() or "document"


def _final_answer(context) -> str:
    if hasattr(context, "final_text"):
        return context.final_text() or ""
    if hasattr(context, "final_answer"):
        return context.final_answer() or ""
    return getattr(context, "answer", None) or getattr(context, "summary", None) or ""


class ChatRequest(BaseModel):
    query: str
    language: str = "auto"
    conversation_id: str = DEFAULT_CONVERSATION_ID


@app.post("/api/chat")
def chat(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        agent = get_agent(request.conversation_id)
        context = agent.run(request.query, language=request.language)
        return {
            "answer": _final_answer(context),
            "sources": getattr(context, "sources", "") or "",
            "stt_text": "",
            "language": context.language,
        }
    except Exception as e:
        log.exception("Chat error")
        raise HTTPException(status_code=500, detail=_error_detail(e))


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    agent = get_agent(request.conversation_id)

    def event_source():
        try:
            if not hasattr(agent, "run_stream"):
                context = agent.run(request.query, language=request.language)
                text = _final_answer(context)
                yield f"data: {json.dumps({'text': text})}\n\n"
                yield "event: done\n"
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "sources": getattr(context, "sources", "") or "",
                            "language": context.language,
                        }
                    )
                    + "\n\n"
                )
                return

            run = agent.run_stream(request.query, language=request.language)
            for chunk in run:
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            context = run.context
            yield "event: done\n"
            yield (
                "data: "
                + json.dumps(
                    {
                        "sources": (context.sources or "") if context else "",
                        "language": context.language if context else request.language,
                    }
                )
                + "\n\n"
            )
        except Exception as e:
            log.exception("Streaming chat error")
            yield f"event: error\ndata: {json.dumps({'detail': _error_detail(e)})}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.post("/api/chat/reset")
def chat_reset(conversation_id: str = DEFAULT_CONVERSATION_ID):
    reset_agent(conversation_id)
    return {"message": "Conversation memory cleared."}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    PDF/image/txt + matching metadata JSON (same base name).

    Folders (unchanged from project-copy):
      ocr/data/ -> OCR -> ocr/output/ -> vector_db -> vector_db/done_docs/
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    docs: dict = {}
    metas: dict = {}

    for f in files:
        name = f.filename or "unknown"
        data = await f.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"File '{name}' is empty.")
        ext = Path(name).suffix.lower()
        stem = _safe_stem(name)
        if ext == ".json":
            metas[stem] = data
        elif ext in {
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg",
            ".tiff",
            ".bmp",
            ".webp",
            ".txt",
            ".docx",
        }:
            docs[stem] = (name, data, ext)
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type '{ext}'. "
                    "Upload PDF/image/txt plus matching .json metadata."
                ),
            )

    if not docs:
        raise HTTPException(
            status_code=400,
            detail="No document files found. Upload documents with their metadata JSON.",
        )

    missing = [stem for stem in docs if stem not in metas]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                "Each document must include a metadata JSON with the same base name. "
                f"Missing JSON for: {', '.join(missing)}"
            ),
        )

    OCR_DATA.mkdir(parents=True, exist_ok=True)
    saved = []
    for stem, (orig_name, data, ext) in docs.items():
        doc_path = OCR_DATA / f"{stem}{ext}"
        json_path = OCR_DATA / f"{stem}.json"
        doc_path.write_bytes(data)
        json_path.write_bytes(metas[stem])
        try:
            json.loads(metas[stem].decode("utf-8"))
        except Exception as e:
            doc_path.unlink(missing_ok=True)
            json_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400, detail=f"Invalid metadata JSON for '{stem}': {e}"
            )
        saved.append(stem)

    try:
        from ocr.main import OCRPipeline
        from vector_db.pipeline import VectorDBPipeline

        OCRPipeline().run_all()
        before = _count_done_docs()
        VectorDBPipeline()._run_ocr_pipeline()
        after = _count_done_docs()
        chunks_added = max(0, after - before)
    except Exception as e:
        log.exception("Document ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {_error_detail(e)}")

    return {
        "message": f"Processed {len(saved)} document(s) successfully.",
        "chunks_added": chunks_added,
        "files": saved,
    }


@app.post("/api/upload/xlsx")
async def upload_xlsx(files: List[UploadFile] = File(...)):
    """
    XLSX + matching metadata JSON (same base name).

    Folders (unchanged from project-copy):
      vector_db/xlsx/<stem>/ -> pipeline -> vector_db/done_xlsx/
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    sheets: dict = {}
    metas: dict = {}

    for f in files:
        name = f.filename or "unknown"
        data = await f.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"File '{name}' is empty.")
        ext = Path(name).suffix.lower()
        stem = _safe_stem(name)
        if ext == ".json":
            metas[stem] = data
        elif ext in {".xlsx", ".xls"}:
            sheets[stem] = (name, data, ext)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Upload .xlsx plus matching .json metadata.",
            )

    if not sheets:
        raise HTTPException(
            status_code=400,
            detail="No Excel files found. Upload .xlsx files with their metadata JSON.",
        )

    missing = [stem for stem in sheets if stem not in metas]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                "Each Excel file must include a metadata JSON with the same base name. "
                f"Missing JSON for: {', '.join(missing)}"
            ),
        )

    XLSX_FOLDER.mkdir(parents=True, exist_ok=True)
    saved = []
    for stem, (orig_name, data, ext) in sheets.items():
        folder = XLSX_FOLDER / stem
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        (folder / f"{stem}{ext}").write_bytes(data)
        (folder / f"{stem}.json").write_bytes(metas[stem])
        try:
            json.loads(metas[stem].decode("utf-8"))
        except Exception as e:
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(
                status_code=400, detail=f"Invalid metadata JSON for '{stem}': {e}"
            )
        saved.append(stem)

    try:
        from vector_db.pipeline import VectorDBPipeline

        before = _count_done_xlsx()
        VectorDBPipeline()._run_excel_pipeline()
        after = _count_done_xlsx()
        chunks_added = max(0, after - before)
    except Exception as e:
        log.exception("Excel ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {_error_detail(e)}")

    return {
        "message": f"Processed {len(saved)} spreadsheet(s) successfully.",
        "chunks_added": chunks_added,
        "files": saved,
    }


@app.get("/api/stored-files")
def stored_files():
    docs_dir = BASE_DIR / "vector_db" / "done_docs"
    xlsx_dir = BASE_DIR / "vector_db" / "done_xlsx"
    docs = sorted(p.name for p in docs_dir.iterdir() if p.is_dir()) if docs_dir.exists() else []
    sheets = sorted(p.name for p in xlsx_dir.iterdir() if p.is_dir()) if xlsx_dir.exists() else []
    return {"documents": docs, "spreadsheets": sheets}


def _count_done_docs() -> int:
    d = BASE_DIR / "vector_db" / "done_docs"
    if not d.exists():
        return 0
    return sum(1 for p in d.iterdir() if p.is_dir())


def _count_done_xlsx() -> int:
    d = BASE_DIR / "vector_db" / "done_xlsx"
    if not d.exists():
        return 0
    return sum(1 for p in d.iterdir() if p.is_dir())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
