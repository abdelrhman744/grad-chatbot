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
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from agent.session import get_agent
from config import settings
from utils import timing

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
            # No fallback to a shared/default id: every client owns one
            # conversation_id for its lifetime (see frontend/lib/conversation.ts)
            # and must send it on every message. Silently defaulting here is
            # exactly the bug that let unrelated conversations merge — see the
            # Issue 2 investigation.
            conversation_id = (payload.get("conversation_id") or "").strip()

            if not conversation_id:
                await websocket.send_json(
                    {"type": "error", "message": "conversation_id is required."}
                )
                continue

            if not query:
                await websocket.send_json({"type": "error", "message": "Query cannot be empty."})
                continue

            await websocket.send_json({"type": "start"})
            await _stream_answer(websocket, query, language, conversation_id)

    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")


async def _stream_answer(websocket: WebSocket, query: str, language: str, conversation_id: str) -> None:
    """
    Timestamp boundaries used for the timing marks below (see
    _log_chat_timing_summary for how they're turned into ms figures):

      - Request start (t=0, RequestTimer.total_ms()'s reference point):
        the very top of this function, before even get_agent() — i.e.
        right after the WebSocket layer handed us a parsed, validated
        chat message. Does NOT include the WebSocket connection's own
        accept()/handshake time, which happens once per connection in
        ws_chat() and is unrelated to any single chat message's latency.
      - "agent_start" mark: the first line inside _produce(), on the
        WORKER THREAD, immediately before agent.run_stream() is invoked.
        The gap between t=0 and this mark is real dispatch/scheduling
        latency (asyncio.to_thread's executor handoff), not agent work.
      - "agent_end" mark: right after agent.run_stream()'s generator is
        fully exhausted (every event produced and hand off to the queue),
        whether it finished normally or raised.
      - "first_token" mark (= TTFT): the first time an item of type
        "token" is dequeued on the event-loop side, about to be handed to
        websocket.send_json() — the earliest point in the EXISTING
        architecture where "this token is ready to go out" can be
        observed without adding new cross-thread signaling. Includes the
        full pipeline up to that point: dispatch + agent planning +
        retrieval + reranking + time-to-first-generated-token, plus the
        (normally sub-millisecond) worker-thread -> asyncio.Queue ->
        event-loop handoff.
      - "stream_end" mark: right after the send loop below finishes
        relaying every queued item (including the final "done"/"error"
        event) — streaming_ms = stream_end - first_token.
    """
    # Lightweight per-MESSAGE correlation id (stdlib uuid4 — no new
    # dependency). conversation_id identifies the CONVERSATION, not this
    # one message, and one conversation can send many messages over the
    # same connection, so a distinct id is needed to unambiguously match
    # one CHAT_TIMING log line to one request when grepping logs.
    request_id = uuid.uuid4().hex[:12]

    # Started here, at the very top (see boundaries above) — BEFORE
    # get_agent() — on the event-loop task, BEFORE the to_thread hop
    # below. asyncio.create_task() captures the current
    # contextvars.Context at creation time, and asyncio.to_thread() copies
    # that same context into its worker thread, so every
    # timing.stage()/substage()/mark() call made deep inside
    # agent.run_stream() (and the nested ThreadPoolExecutors in
    # rag_service.py, via run_concurrent_ctx()) lands on this same
    # RequestTimer instead of silently no-op'ing. Mirrors the pattern
    # already used by POST /api/chat (routes/chat.py) — see utils/timing.py.
    t = None
    if settings.LOG_REQUEST_PROFILE:
        t = timing.start(
            f"WS /ws/chat request_id={request_id} conversation={conversation_id!r} query={query[:60]!r}"
        )

    agent = get_agent(conversation_id)
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _produce() -> None:
        if t is not None:
            t.mark("agent_start")
        try:
            for event in agent.run_stream(query, language=language):
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as e:  # noqa: BLE001 - surfaced to the client below
            log.exception("Agent streaming error")
            loop.call_soon_threadsafe(
                queue.put_nowait, {"type": "error", "message": f"{type(e).__name__}: {e}"}
            )
        finally:
            if t is not None:
                t.mark("agent_end")
            loop.call_soon_threadsafe(queue.put_nowait, None)

    producer = asyncio.create_task(asyncio.to_thread(_produce))

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            if t is not None and item.get("type") == "token" and "first_token" not in t.marks:
                t.mark("first_token")
            await websocket.send_json(item)

        if t is not None:
            t.mark("stream_end")

        await producer
    finally:
        # Always finish/log the timer, even if the websocket send loop
        # raises (e.g. the client disconnects mid-stream) — an in-flight
        # RequestTimer left in the contextvar would otherwise leak into
        # whatever this Task's context touches next. Runs on every exit
        # path (success, agent error, or an exception here), so partial
        # timing is still logged on failure — this never swallows or
        # replaces the original exception, which keeps propagating
        # normally after this finally block.
        if settings.LOG_REQUEST_PROFILE and t is not None:
            report = timing.finish()
            if report:
                log.info("\n" + report)
            _log_chat_timing_summary(request_id, t)


def _log_chat_timing_summary(request_id: str, t: timing.RequestTimer) -> None:
    """
    One concise, grep-friendly CHAT_TIMING summary per chat message, in
    ADDITION to (not instead of) the detailed [profile] report already
    logged from timing.finish() — same RequestTimer/data (t.notes /
    t.marks), just a flatter, fixed-field view suited to log
    aggregation/alerting. See _stream_answer's docstring for the exact
    timestamp boundaries behind ttft_ms/streaming_ms/dispatch_ms/agent_ms.

    Every field is read from data the SAME timing utility already
    collected — no second/competing timing mechanism, and no field here
    is invented if the underlying stage/substage/mark was never recorded
    (e.g. an agent that never retrieves has no retrieval_ms; a request
    that errors before its first token has no ttft_ms) — those show as
    "n/a" rather than a fabricated 0.0.
    """
    notes = t.notes
    marks = t.marks

    agent_start = marks.get("agent_start")
    agent_end = marks.get("agent_end")
    agent_ms = agent_end - agent_start if agent_start is not None and agent_end is not None else None

    first_token = marks.get("first_token")
    stream_end = marks.get("stream_end")
    streaming_ms = (
        stream_end - first_token if first_token is not None and stream_end is not None else None
    )

    def fmt(v: float | None) -> str:
        return f"{v:.1f}" if v is not None else "n/a"

    log.info(
        "CHAT_TIMING\n"
        f"    request_id      = {request_id}\n"
        f"    total_ms        = {fmt(t.total_ms())}\n"
        f"    dispatch_ms     = {fmt(agent_start)}\n"
        f"    agent_ms        = {fmt(agent_ms)}\n"
        f"    planner_ms      = {fmt(notes.get('agent_planning'))}\n"
        f"    retrieval_ms    = {fmt(notes.get('retrieval_total_ms'))}\n"
        f"    rewrite_ms      = {fmt(notes.get('query_rewrite_ms'))}\n"
        f"    translate_ms    = {fmt(notes.get('translate_ms'))}\n"
        f"    embedding_ms    = {fmt(notes.get('embedding_ms'))}\n"
        f"    qdrant_ms       = {fmt(notes.get('qdrant_search_ms'))}\n"
        f"    rerank_ms       = {fmt(notes.get('reranking'))}\n"
        f"    mmr_ms          = {fmt(notes.get('mmr_diversification'))}\n"
        f"    generation_ms   = {fmt(notes.get('llm_generation'))}\n"
        f"    ttft_ms         = {fmt(first_token)}\n"
        f"    streaming_ms    = {fmt(streaming_ms)}"
    )
