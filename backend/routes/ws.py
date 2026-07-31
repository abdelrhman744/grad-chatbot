"""
ws.py

WebSocket endpoint that streams the agent's final answer token-by-token,
the same way ChatGPT-style UIs render responses, instead of waiting for
the full answer and returning it in one HTTP response like /api/chat does.

Protocol (JSON text frames both ways):

  Client -> Server (send once per question):
    {"query": "...", "language": "auto" | "ar" | "en", "conversation_id": "..."}

  Server -> Client (one connection can be reused for many questions):
    {"type": "start"}                                   # question accepted
    {"type": "token", "text": "..."}                     # repeated, in order
    {"type": "done", "answer": "...", "sources": "..."}  # final message
    {"type": "error", "message": "..."}                  # on failure

The agent's underlying LLM calls are synchronous (the `groq` SDK is not
async), so `agent.run_stream()` is driven on a worker thread and its
yielded events are relayed to the client through an asyncio.Queue — this
keeps the event loop free to serve other connections while a completion
streams in.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from agent.session import get_agent
from config import settings

log = logging.getLogger("routes.ws")

router = APIRouter()


@router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON message."})
                continue

            query = (payload.get("query") or "").strip()
            language = payload.get("language", "auto")
            conversation_id = payload.get("conversation_id", settings.DEFAULT_CONVERSATION_ID)

            if not query:
                await websocket.send_json({"type": "error", "message": "Query cannot be empty."})
                continue

            await websocket.send_json({"type": "start"})
            await _stream_answer(websocket, query, language, conversation_id)

    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")


async def _stream_answer(websocket: WebSocket, query: str, language: str, conversation_id: str) -> None:
    agent = get_agent(conversation_id)
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _produce() -> None:
        try:
            for event in agent.run_stream(query, language=language):
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as e:  # noqa: BLE001 - surfaced to the client below
            log.exception("Agent streaming error")
            loop.call_soon_threadsafe(
                queue.put_nowait, {"type": "error", "message": f"{type(e).__name__}: {e}"}
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    producer = asyncio.create_task(asyncio.to_thread(_produce))

    while True:
        item = await queue.get()
        if item is None:
            break
        await websocket.send_json(item)

    await producer
