import asyncio
import logging

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Literal

from agent.session import get_agent, reset_agent
from config import settings
from services.audio_service import transcribe_audio
from services.rag_service import build_sources_from_dicts, delete_conversation_documents
from utils import timing
from utils.conversation_auth import (
    new_conversation_id,
    require_valid_conversation_token,
    sign_conversation_id,
)

log = logging.getLogger("routes.chat")

router = APIRouter()


def _error_detail(e: Exception) -> str:
    """
    Build a non-empty, informative error message for the HTTP response.
    Some exceptions (e.g. certain SDK/network errors) stringify to an
    empty string, which would otherwise surface to the frontend as a bare
    "Internal Server Error" with no actionable information. Always include
    the exception type so the real cause is visible in the response body,
    not just in the server logs.
    """
    msg = str(e).strip()
    return f"{type(e).__name__}: {msg}" if msg else type(e).__name__


class ChatRequest(BaseModel):
    query: str
    language: Literal["auto", "ar", "en"] = "auto"
    # No default: every caller must own and send a real conversation_id (see
    # frontend/lib/conversation.ts). A default here is exactly the silent
    # fallback that let unrelated conversations merge — see Issue 2.
    conversation_id: str
    # Proof that the caller actually owns conversation_id — see
    # utils/conversation_auth.py. Obtained once from POST /api/session.
    conversation_token: str


@router.post("/session")
async def create_session():
    """
    Mint a fresh conversation_id and its signature. The frontend calls
    this once per tab (see frontend/lib/conversation.ts) instead of
    generating conversation_id itself — the server must be the only party
    that ever creates one, since a signature is only meaningful for an id
    the server generated (see utils/conversation_auth.py's docstring).
    """
    conversation_id = new_conversation_id()
    return {"conversation_id": conversation_id, "conversation_token": sign_conversation_id(conversation_id)}


def _run_agent(query: str, language: str, conversation_id: str) -> dict:
    agent = get_agent(conversation_id)
    context = agent.run(query, language=language)

    answer = context.final_answer() or (
        "المعلومة غير موجودة في الملفات المرفوعة."
        if context.language == "ar"
        else "The information is not available in the uploaded files."
    )

    return {
        "answer": answer,
        "sources": build_sources_from_dicts(context.documents, lang=context.language),
        "report": context.report,
    }


@router.post("/chat")
async def chat(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Started here (on the event-loop thread) and picked up via contextvars
    # by asyncio.to_thread below, so every stage() call anywhere in the
    # agent/rag_service/memory pipeline for this request lands on the same
    # RequestTimer — see utils/timing.py.
    if settings.LOG_REQUEST_PROFILE:
        timing.start(f"POST /api/chat conversation={request.conversation_id!r} query={request.query[:60]!r}")
    try:
        # The agent's LLM calls are synchronous (blocking) — run them on a
        # worker thread so one slow request doesn't stall the event loop
        # for every other concurrent user, matching the pattern /ws/chat
        # already uses (routes/ws.py).
        result = await asyncio.to_thread(
            _run_agent, request.query, request.language, request.conversation_id
        )
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "stt_text": "",
            "report": result["report"],
        }
    except Exception as e:
        log.exception("Agent chat error")
        raise HTTPException(status_code=500, detail=_error_detail(e))
    finally:
        report = timing.finish()
        if report:
            log.info("\n" + report)


@router.post("/chat/voice")
async def chat_voice(
    audio: UploadFile = File(...),
    language: str = Form("auto"),
    # No default — see ChatRequest.conversation_id above.
    conversation_id: str = Form(...),
):
    """Transcribe uploaded audio then answer the question via the agent."""
    try:
        audio_bytes = await audio.read()
        lang_hint = language if language in {"ar", "en"} else None
        # transcribe_audio() is synchronous, CPU/IO-heavy work (ffmpeg
        # subprocess + Whisper decoding) — run it on a worker thread so it
        # doesn't block the event loop for every other concurrent request,
        # matching the asyncio.to_thread(_run_agent, ...) pattern already
        # used below and in POST /chat.
        stt_text = await asyncio.to_thread(transcribe_audio, audio_bytes, language=lang_hint)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Audio transcription failed: {e}")

    if not stt_text:
        raise HTTPException(
            status_code=422,
            detail="Whisper could not recognise any speech. Check microphone / audio quality.",
        )

    try:
        result = await asyncio.to_thread(_run_agent, stt_text, language, conversation_id)
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "stt_text": stt_text,
            "report": result["report"],
        }
    except Exception as e:
        log.exception("Agent voice chat error")
        raise HTTPException(status_code=500, detail=_error_detail(e))


@router.post("/chat/reset")
async def chat_reset(conversation_id: str):
    """
    Clear short-term/long-term memory AND this conversation's indexed
    documents. No default — resets exactly the one conversation_id
    supplied, never a shared/fallback id (see ChatRequest.conversation_id
    above).

    Document removal is a REAL Qdrant deletion (see
    rag_service.delete_conversation_documents), scoped by the same
    conversation_id filter used at retrieval time, so it can only ever
    affect this conversation_id's own chunks — see Document Isolation.
    """
    reset_agent(conversation_id)
    documents_removed = delete_conversation_documents(conversation_id)
    return {
        "message": "Conversation memory and documents cleared.",
        "documents_removed": documents_removed,
    }
