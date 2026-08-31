"""
agent.py

Main ReAct agent. At each step it decides whether to retrieve documents,
generate an answer, summarize, compare, or respond directly from memory,
executes that single action, and loops until a terminal action is reached
or max_iterations is hit.

One Agent instance owns one conversation's memory (see agent/session.py
for the per-conversation registry used by the API layer).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import Counter
from contextlib import contextmanager

from config import settings
from memory.memory_manager import MemoryManager
from services import rag_service
from services.rag_service import detect_language, build_sources_from_dicts

from utils import timing

from .llm import AgentLLM
from .prompt import SYSTEM_PROMPT, USER_PROMPT
from .registry import build_tools
from .schemas import ExecutionContext, RetrieveAction, RetrieveArguments, ToolName, TERMINAL_TOOLS

log = logging.getLogger("agent")

# Deterministic (no Groq call) check for the one case the planner's HARD
# RULE (see prompt.py) allows "respond"/"generate" without a prior
# "retrieve" this turn: a message that is PURELY a greeting, thanks,
# farewell, or a question about the conversation itself. Used by
# `Agent._correct_premature_terminal` below instead of re-asking the LLM —
# see that method's docstring for why the second Groq round-trip was
# removed.
#
# Matching is a whole-message, word-count-gated phrase lookup (not a
# "starts with" or "contains" check): a message longer than
# _SMALL_TALK_MAX_WORDS never matches, however it starts, so "hi, what's
# the deadline in the contract?" still falls through to a forced
# "retrieve" — only genuinely short, pure small-talk messages are trusted.
_SMALL_TALK_MAX_WORDS = 6

_SMALL_TALK_PHRASES = {
    # English greetings / thanks / farewells / filler acknowledgements
    "hi", "hello", "hi there", "hello there", "hey", "hey there", "yo", "sup",
    "good morning", "good afternoon", "good evening", "good night",
    "how are you", "hows it going", "how's it going", "what's up", "whats up",
    "thanks", "thank you", "thanks a lot", "thank you so much", "thx", "ty",
    "appreciate it", "much appreciated",
    "bye", "goodbye", "see you", "see you later", "take care", "good night",
    "ok", "okay", "cool", "great", "nice", "got it", "sounds good", "awesome",
    "who are you", "what can you do", "what is your name", "whats your name",
    "what did i ask", "what did i ask you", "what did i just ask",
    "what did i ask before", "what did i ask you before",
    "what did you say", "can you repeat that", "can you repeat",
    # Arabic equivalents
    "مرحبا", "اهلا", "أهلا", "هاي", "السلام عليكم",
    "صباح الخير", "مساء الخير",
    "ازيك", "عامل ايه", "عاملة ايه", "كيف حالك", "كيفك",
    "شكرا", "متشكر", "متشكرة", "يسلمو", "تسلم", "تسلمي", "الله يسلمك",
    "باي", "مع السلامة", "تصبح على خير",
    "تمام", "حلو", "كويس", "اوك", "ماشي",
    "مين انت", "اسمك ايه", "تقدر تعمل ايه",
    "ايه اللي سألتك", "قولتلك ايه", "تقدر تعيد",
}


def _looks_like_small_talk(question: str) -> bool:
    """No-LLM-call check: strip trailing punctuation/whitespace, collapse
    internal whitespace, lowercase, and look up the whole normalized
    message in `_SMALL_TALK_PHRASES` — but only if it's short enough
    (`_SMALL_TALK_MAX_WORDS`) to plausibly be pure small talk at all."""
    text = (question or "").strip()
    if not text:
        return False
    stripped = re.sub(r"[\s!.,?؟،]+$", "", text).strip()
    if not stripped or len(stripped.split()) > _SMALL_TALK_MAX_WORDS:
        return False
    normalized = re.sub(r"\s+", " ", stripped).strip().lower()
    return normalized in _SMALL_TALK_PHRASES


class Agent:
    def __init__(self, conversation_id: str = settings.DEFAULT_CONVERSATION_ID):
        self.conversation_id = conversation_id

        self.llm = AgentLLM()

        self.memory_manager = MemoryManager(conversation_id=conversation_id)

        # The document this conversation is currently "about" — updated
        # whenever a retrieval step returns chunks from a document, or a
        # report is generated for one. Lets "اعمل تقرير" / "generate a
        # report" resolve which file to use without the user repeating
        # its name, as long as there's a document already in play.
        self.active_document: str | None = None

        # Rendered once per turn, at the top of _run_impl/_run_stream_impl,
        # and read from here everywhere else in that turn (planning prompt
        # on every ReAct iteration, terminal-tool execution, the
        # max-iterations fallback). Previously each of those call sites
        # called self.memory_manager.as_prompt_text() independently, so a
        # single turn that took N planning iterations re-rendered and
        # re-sent the same memory text N+1 times — this cache makes it
        # exactly once per turn regardless of how many iterations the
        # ReAct loop takes.
        self._current_memory_text: str = ""

        self.tools = build_tools(
            memory_text_provider=lambda: self._current_memory_text,
            conversation_id=conversation_id,
            active_document_provider=lambda: self.active_document,
            active_document_setter=self._set_active_document,
        )

        self.max_iterations = settings.AGENT_MAX_ITERATIONS

        # ── Lifecycle bookkeeping (see agent/session.py's cleanup loop) ────
        # `last_active`/`_in_flight` are read by session.py's background
        # cleanup to decide whether this conversation's Agent is safe to
        # evict from the in-process registry. Guarded by their own lock
        # (not the registry's) since they're mutated from whichever request
        # thread happens to be running this Agent, independent of registry
        # lookups happening for OTHER conversation_ids at the same time.
        self.last_active: float = time.monotonic()
        self._activity_lock = threading.Lock()
        self._in_flight = 0

    def touch(self) -> None:
        """Record activity now. Called by agent/session.py on every lookup
        so a conversation that's merely being looked up (not necessarily
        mid-`run`) still counts as recently active."""
        with self._activity_lock:
            self.last_active = time.monotonic()

    def is_idle(self, timeout_seconds: float) -> bool:
        """True only if this Agent has NO in-flight request AND has been
        inactive for at least `timeout_seconds`. An Agent currently inside
        `run()`/`run_stream()` is never idle, no matter how long ago
        `last_active` was set or how long the request is taking (e.g. a
        Groq rate-limit retry storm) — see `_mark_busy`."""
        with self._activity_lock:
            if self._in_flight > 0:
                return False
            return (time.monotonic() - self.last_active) >= timeout_seconds

    @contextmanager
    def _mark_busy(self):
        """Wraps the full duration of one `run()`/`run_stream()` call so
        `is_idle()` can never return True for an Agent currently handling a
        request, regardless of how the cleanup loop's timing lines up
        against a slow request."""
        with self._activity_lock:
            self._in_flight += 1
            self.last_active = time.monotonic()
        try:
            yield
        finally:
            with self._activity_lock:
                self._in_flight -= 1
                self.last_active = time.monotonic()

    def _set_active_document(self, filename: str) -> None:
        self.active_document = filename

    def _update_active_document_from_retrieval(self, context: ExecutionContext) -> None:
        """After a retrieve step, remember whichever document most of the
        retrieved chunks came from as the conversation's active document."""
        sources = [
            d.get("metadata", {}).get("source")
            for d in context.documents
            if d.get("metadata", {}).get("source")
        ]
        if not sources:
            return
        top_source = Counter(sources).most_common(1)[0][0]
        self.active_document = top_source

    # ── Public API ───────────────────────────────────────────────────────

    def run(self, question: str, language: str = "auto", debug: bool = False) -> ExecutionContext:
        with self._mark_busy():
            return self._run_impl(question, language=language, debug=debug)

    def _run_impl(self, question: str, language: str = "auto", debug: bool = False) -> ExecutionContext:
        with timing.stage("language_detection"):
            detected_lang = detect_language(question) if language == "auto" else language
        context = ExecutionContext(language=detected_lang)

        with timing.stage("memory_loading"):
            self._current_memory_text = self.memory_manager.as_prompt_text()

        for iteration in range(self.max_iterations):
            messages = self._build_messages(question, context)
            with timing.stage("agent_planning"):
                action = self.llm.invoke(messages, fallback_question=question)
                action = self._correct_premature_terminal(messages, action, context, question)

            if debug or settings.AGENT_DEBUG:
                self._debug_step(iteration + 1, action)

            if action.action == ToolName.RETRIEVE:
                current_question = action.arguments.question.strip().lower()
                previous_questions = [q.lower() for q in context.retrieved_questions]

                if current_question in previous_questions:
                    context.observations.append({
                        "tool": "retrieve",
                        "question": action.arguments.question,
                        "status": "already_retrieved",
                    })
                    continue

                context = self._run_tool(action, context, question)
                context.retrieved_questions.append(action.arguments.question)
                self._update_active_document_from_retrieval(context)
                continue

            context = self._run_tool(action, context, question)

            if action.action in TERMINAL_TOOLS:
                break
        else:
            # Exhausted max_iterations without reaching a terminal action.
            # Force a final answer from whatever context we have so the
            # user always gets a response.
            log.warning(f"Agent hit max_iterations ({self.max_iterations}) without finishing.")
            context = self.tools["generate"].run(context, question=question)

        self._remember(question, context)

        return context

    def reset_memory(self) -> None:
        self.memory_manager.reset()

    # ── Streaming public API (used by /ws/chat) ─────────────────────────────
    #
    # Runs the exact same ReAct loop as `run()` for every non-terminal step
    # (retrieval, etc — these are internal reasoning and aren't shown to the
    # user token-by-token). Once the agent picks a *terminal* action, instead
    # of calling that tool's `run()` (which invokes the LLM synchronously and
    # returns the full string), this streams the underlying Groq completion
    # chunk by chunk so the frontend can render it as it's generated.
    #
    # Yields dicts: {"type": "token", "text": "..."} for each chunk, then
    # exactly one {"type": "done", "answer": "...", "sources": "..."} at the
    # end once the full answer has been produced and memory updated.

    def run_stream(self, question: str, language: str = "auto"):
        with self._mark_busy():
            yield from self._run_stream_impl(question, language=language)

    def _run_stream_impl(self, question: str, language: str = "auto"):
        with timing.stage("language_detection"):
            detected_lang = detect_language(question) if language == "auto" else language
        context = ExecutionContext(language=detected_lang)
        full_text = ""

        with timing.stage("memory_loading"):
            self._current_memory_text = self.memory_manager.as_prompt_text()

        for iteration in range(self.max_iterations):
            messages = self._build_messages(question, context)
            with timing.stage("agent_planning"):
                action = self.llm.invoke(messages, fallback_question=question)
                action = self._correct_premature_terminal(messages, action, context, question)

            if settings.AGENT_DEBUG:
                self._debug_step(iteration + 1, action)

            if action.action == ToolName.RETRIEVE:
                current_question = action.arguments.question.strip().lower()
                previous_questions = [q.lower() for q in context.retrieved_questions]

                if current_question in previous_questions:
                    context.observations.append({
                        "tool": "retrieve",
                        "question": action.arguments.question,
                        "status": "already_retrieved",
                    })
                    continue

                context = self._run_tool(action, context, question)
                context.retrieved_questions.append(action.arguments.question)
                self._update_active_document_from_retrieval(context)
                continue

            if action.action in TERMINAL_TOOLS:
                if action.action == ToolName.REPORT:
                    # Report generation is a slow, multi-step pipeline
                    # (re-read the document, map-reduce with the LLM,
                    # render the PDF) — there's no meaningful token
                    # stream for it, so emit a status update instead and
                    # run it to completion on this same worker thread.
                    yield {
                        "type": "status",
                        "text": (
                            "جاري تجهيز التقرير، ده ممكن ياخد شوية وقت..."
                            if detected_lang == "ar"
                            else "Generating the report — this may take a moment..."
                        ),
                    }
                    context = self._run_tool(action, context, question)
                    full_text = context.answer or ""
                else:
                    # Wraps the full token-by-token generation, same stage
                    # name/semantics as generate_answer()'s
                    # timing.stage("llm_generation") on the non-streaming
                    # path (rag_service.py) — measures end-to-end LLM
                    # completion time here since there is no single
                    # invoke() call to wrap around a streaming generator.
                    with timing.stage("llm_generation"):
                        for piece in self._stream_terminal_action(action, context):
                            full_text += piece
                            yield {"type": "token", "text": piece}
                    self._finalize_stream(action, context, full_text)
                break

            context = self._run_tool(action, context, question)
        else:
            # Exhausted max_iterations — force a streamed final answer from
            # whatever context we have so the user always gets a response.
            log.warning(f"Agent hit max_iterations ({self.max_iterations}) without finishing.")
            with timing.stage("llm_generation"):
                for piece in rag_service.generate_answer_stream(
                    question, context.documents, lang=context.language, memory=self._current_memory_text
                ):
                    full_text += piece
                    yield {"type": "token", "text": piece}
            context.answer = full_text.strip()

        self._remember(question, context)

        done_event = {
            "type": "done",
            "answer": context.final_answer() or "",
            "sources": build_sources_from_dicts(context.documents, lang=context.language),
        }
        if context.report:
            done_event["report"] = context.report

        yield done_event

    def _stream_terminal_action(self, action, context: ExecutionContext):
        """Dispatch a terminal action to its streaming rag_service function.
        Mirrors the logic in agent/tools/*.py, but yields text chunks
        instead of returning a finished string."""
        memory_text = self._current_memory_text

        if action.action == ToolName.GENERATE:
            question = action.arguments.question
            if not context.documents:
                if memory_text:
                    yield from rag_service.answer_from_memory_stream(
                        question, memory_text, lang=context.language
                    )
                else:
                    yield (
                        "لا توجد مستندات كافية للإجابة على هذا السؤال."
                        if context.language == "ar"
                        else "No relevant documents were found to answer this question."
                    )
            else:
                yield from rag_service.generate_answer_stream(
                    question, context.documents, lang=context.language, memory=memory_text
                )

        elif action.action == ToolName.SUMMARIZE:
            yield from rag_service.summarize_stream(context.documents, lang=context.language)

        elif action.action == ToolName.COMPARE:
            yield from rag_service.compare_stream(
                action.arguments.question, context.documents, lang=context.language
            )

        elif action.action == ToolName.RESPOND:
            yield from rag_service.answer_from_memory_stream(
                action.arguments.question, memory_text, lang=context.language
            )

    @staticmethod
    def _finalize_stream(action, context: ExecutionContext, full_text: str) -> None:
        """Write the accumulated streamed text into the same ExecutionContext
        field its non-streaming tool counterpart would have set."""
        text = full_text.strip()
        if action.action == ToolName.SUMMARIZE:
            context.summary = text
        elif action.action == ToolName.COMPARE:
            context.comparison = text
        else:  # GENERATE or RESPOND
            context.answer = text
        context.observations.append({"tool": action.action.value, "status": "streamed"})

    # ── Internal ─────────────────────────────────────────────────────────

    def _build_messages(self, question: str, context: ExecutionContext) -> list[dict]:
        # memory_text is computed once per turn (see _run_impl /
        # _run_stream_impl), not recomputed on every ReAct iteration —
        # this method just reads the cached value.
        memory_text = self._current_memory_text or "(none)"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    question=question,
                    active_document=self.active_document or "(none)",
                    memory=memory_text,
                    documents=len(context.documents),
                    observations=self._format_observations(context.observations),
                    retrieved_questions=", ".join(context.retrieved_questions) or "(none)",
                ),
            },
        ]

    @staticmethod
    def _format_observations(observations: list[dict]) -> str:
        if not observations:
            return "(none)"
        return "\n".join(json.dumps(o, ensure_ascii=False) for o in observations)

    def _run_tool(self, action, context: ExecutionContext, raw_question: str = "") -> ExecutionContext:
        tool = self.tools[action.action.value]
        kwargs = action.arguments.model_dump()
        if action.action == ToolName.RETRIEVE and not context.retrieved_questions:
            # Anchor only the FIRST retrieve of a turn to the user's own
            # literal wording, in addition to whatever the planner LLM
            # chose to search for. The planner is free (and needs to be
            # free) to reformulate the retrieve query for coreference
            # resolution on LATER retrieves this turn (see prompt.py's
            # Reference & Coreference Resolution section), but that
            # reformulation is not guaranteed to be stable run-to-run for a
            # hosted LLM even at temperature 0 (see
            # _correct_premature_terminal's docstring) — confirmed directly:
            # the exact same first-retrieve user question can make the
            # planner emit different search text on different runs, which
            # changes the retrieval candidate pool even though the
            # underlying retrieval pipeline itself is fully deterministic
            # for a fixed query string. See rag_service._retrieve's
            # `raw_question` parameter — this adds zero extra LLM calls.
            kwargs["raw_question"] = raw_question
        return tool.run(context=context, **kwargs)

    def _correct_premature_terminal(self, messages: list[dict], action, context: ExecutionContext, question: str):
        """
        Deterministic backstop for a routing mistake the planner makes
        intermittently (confirmed via direct testing — see Issue 2
        investigation): choosing "respond" or "generate" WITHOUT ever
        calling "retrieve" this turn, for a message that plainly asks for
        document content (most often reproduced with short, imperative
        phrasings, e.g. Arabic "اشرح لي X" / English "Explain X" — no
        question mark, and easy for the small, low-latency AGENT_MODEL to
        misclassify as small talk despite the system prompt's explicit
        "HARD RULE" against it). Same exact input, temperature 0, has been
        observed to occasionally choose "retrieve" and occasionally not —
        hosted LLM inference is not perfectly reproducible run-to-run — so
        prompting alone cannot guarantee the rule; this enforces it in code.

        Previously this re-asked the SAME planner model a second time over
        the network to double-check itself — an extra, unconditional Groq
        round-trip on every occurrence, and a real contributor to the
        excessive-Groq-calls / 429 issue (see Issue 1 investigation).
        Asking an LLM to compensate for another LLM call's mistake doesn't
        need a third network round-trip: the only question that second
        call was ever really answering — "is this message pure small talk,
        or does it need document lookup?" — is answered locally instead,
        via `_looks_like_small_talk`. If the message matches, the
        planner's original "respond"/"generate" choice is trusted as-is
        (zero extra calls). Otherwise the action is deterministically
        forced to "retrieve" using the raw question (also zero extra
        calls) — this is strictly a stronger guarantee of the HARD RULE
        than the old re-ask, which could (and per its own docstring,
        occasionally did) still return "respond" a second time.

        Only fires once per turn, only when nothing has been retrieved yet.
        """
        if action.action not in (ToolName.RESPOND, ToolName.GENERATE):
            return action
        if context.retrieved_questions or context.documents:
            return action

        if _looks_like_small_talk(question):
            return action

        log.warning(
            f"Agent chose '{action.action.value}' on iteration 1 with nothing retrieved yet "
            f"for question={question!r} — deterministically forcing 'retrieve' per the "
            "prompt's own HARD RULE (no extra Groq call)."
        )
        return RetrieveAction(
            thought=(
                "Deterministic override: previous action skipped retrieval for a "
                "non-small-talk message, which violates the HARD RULE."
            ),
            action=ToolName.RETRIEVE,
            arguments=RetrieveArguments(question=question[:500], top_k=5),
        )

    def _remember(self, question: str, context: ExecutionContext) -> None:
        answer = context.final_answer()
        if answer:
            with timing.stage("memory_persist"):
                self.memory_manager.add_turn(user_message=question, assistant_message=answer)

    def _debug_step(self, iteration: int, action) -> None:
        log.debug(f"--- iteration {iteration} ---")
        log.debug(f"thought: {action.thought}")
        log.debug(f"action: {action.action.value}")
        log.debug(f"arguments: {action.arguments.model_dump()}")
