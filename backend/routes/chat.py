import logging

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Literal

from agent.session import get_agent, reset_agent
from config import settings
from services.audio_service import transcribe_audio
from services.rag_service import build_sources_from_dicts

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
    conversation_id: str = settings.DEFAULT_CONVERSATION_ID


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
    try:
        result = _run_agent(request.query, request.language, request.conversation_id)
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "stt_text": "",
            "report": result["report"],
        }
    except Exception as e:
        log.exception("Agent chat error")
        raise HTTPException(status_code=500, detail=_error_detail(e))


@router.post("/chat/voice")
async def chat_voice(
    audio: UploadFile = File(...),
    language: str = Form("auto"),
    conversation_id: str = Form(settings.DEFAULT_CONVERSATION_ID),
):
    """Transcribe uploaded audio then answer the question via the agent."""
    try:
        audio_bytes = await audio.read()
        lang_hint = language if language in {"ar", "en"} else None
        stt_text = transcribe_audio(audio_bytes, language=lang_hint)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Audio transcription failed: {e}")

    if not stt_text:
        raise HTTPException(
            status_code=422,
            detail="Whisper could not recognise any speech. Check microphone / audio quality.",
        )

    try:
        result = _run_agent(stt_text, language, conversation_id)
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
async def chat_reset(conversation_id: str = settings.DEFAULT_CONVERSATION_ID):
    """Clear short-term and long-term memory for a conversation."""
    reset_agent(conversation_id)
    return {"message": "Conversation memory cleared."}
