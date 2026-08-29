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
from memory.llm_adapter import LLMTextGenerator
from memory.memory_manager import MemoryManager
from services import rag_service
from services.rag_service import detect_language, build_sources_from_dicts

from utils import timing

from .llm import AgentLLM
from .prompt import SYSTEM_PROMPT, USER_PROMPT
from .registry import build_tools
from .schemas import (
    ExecutionContext,
    GenerateAction,
    GenerateArguments,
    RetrieveAction,
    RetrieveArguments,
    ToolName,
    TERMINAL_TOOLS,
)

log = logging.getLogger("agent")

# Deterministic (no Groq call) check for the one case the planner's HARD
# RULE (see prompt.py) allows "respond"/"generate" without a prior
# "retrieve" this turn: a message that is PURELY a greeting, thanks,
# farewell, or a question about the conversation itself. Used by
# `Agent._correct_premature_terminal` below instead of re-asking the LLM —
# see that method's docstring for why the second Groq round-trip was
# removed — AND by the pre-planner fast path below (`_classify_fast_path`)
# to skip the planner call entirely for this same class of message.
#
# Matching is a whole-message, word-count-gated phrase lookup (not a
# "starts with" or "contains" check): a message longer than
# _SMALL_TALK_MAX_WORDS never matches, however it starts, so "hi, what's
# the deadline in the contract?" still falls through to a forced
# "retrieve" — only genuinely short, pure small-talk messages are trusted.
_SMALL_TALK_MAX_WORDS = 6

# Pure greeting/thanks/farewell/acknowledgement/identity phrases have a
# fixed, correct answer that doesn't depend on conversation content at
# all — these get a CANNED response with NO Groq call whatsoever (see
# `_classify_fast_path`). Split by category only so each gets a
# category-appropriate canned reply; the categories carry no other meaning.
_CANNED_PHRASES: dict[str, set[str]] = {
    "greeting": {
        "hi", "hello", "hi there", "hello there", "hey", "hey there", "yo", "sup",
        "good morning", "good afternoon", "good evening", "good night",
        "how are you", "hows it going", "how's it going", "what's up", "whats up",
        "مرحبا", "اهلا", "أهلا", "هاي", "السلام عليكم",
        "صباح الخير", "مساء الخير",
        "ازيك", "عامل ايه", "عاملة ايه", "كيف حالك", "كيفك",
    },
    "thanks": {
        "thanks", "thank you", "thanks a lot", "thank you so much", "thx", "ty",
        "appreciate it", "much appreciated",
        "شكرا", "متشكر", "متشكرة", "يسلمو", "تسلم", "تسلمي", "الله يسلمك",
    },
    "farewell": {
        "bye", "goodbye", "see you", "see you later", "take care",
        "باي", "مع السلامة", "تصبح على خير",
    },
    "ack": {
        "ok", "okay", "cool", "great", "nice", "got it", "sounds good", "awesome",
        "تمام", "حلو", "كويس", "اوك", "ماشي",
    },
    "identity": {
        "who are you", "what can you do", "what is your name", "whats your name",
        "مين انت", "اسمك ايه", "تقدر تعمل ايه",
    },
}

# Meta-conversation questions genuinely need the conversation's own memory
# content to answer correctly ("what did I just ask?") — these still skip
# the PLANNER call (they're unambiguously small talk, never document
# lookup), but still need one real generation call over memory text. See
# `_classify_fast_path`.
_MEMORY_SMALL_TALK_PHRASES = {
    "what did i ask", "what did i ask you", "what did i just ask",
    "what did i ask before", "what did i ask you before",
    "what did you say", "can you repeat that", "can you repeat",
    "ايه اللي سألتك", "قولتلك ايه", "تقدر تعيد",
}

_SMALL_TALK_PHRASES = frozenset(
    {p for phrases in _CANNED_PHRASES.values() for p in phrases} | _MEMORY_SMALL_TALK_PHRASES
)

# One fixed reply per canned category (EN/AR) — used verbatim, no LLM call.
_CANNED_RESPONSES: dict[str, dict[str, str]] = {
    "greeting": {
        "en": "Hello! How can I help you with your documents today?",
        "ar": "أهلاً بك! إزاي أقدر أساعدك في مستنداتك النهاردة؟",
    },
    "thanks": {
        "en": "You're welcome! Let me know if there's anything else you'd like to explore in your documents.",
        "ar": "العفو! قوللي لو حابب تستكشف حاجة تانية في مستنداتك.",
    },
    "farewell": {
        "en": "Goodbye! Feel free to come back anytime you have questions about your documents.",
        "ar": "مع السلامة! ارجعلي في أي وقت لو كان عندك أسئلة عن مستنداتك.",
    },
    "ack": {
        "en": "Got it! Let me know if you have any questions about your documents.",
        "ar": "تمام! قوللي لو عندك أي سؤال عن مستنداتك.",
    },
    "identity": {
        "en": "I'm an AI assistant that answers questions strictly from the documents you've uploaded — ask me anything about their content.",
        "ar": "أنا مساعد ذكاء اصطناعي بجاوب على الأسئلة بالاعتماد فقط على المستندات اللي رفعتها — اسألني عن أي حاجة فيها.",
    },
}


def _normalize_small_talk(question: str) -> str:
    """Strip trailing punctuation/whitespace, collapse internal whitespace,
    lowercase. Returns "" if the message is empty or too long
    (`_SMALL_TALK_MAX_WORDS`) to plausibly be pure small talk at all."""
    text = (question or "").strip()
    if not text:
        return ""
    stripped = re.sub(r"[\s!.,?؟،]+$", "", text).strip()
    if not stripped or len(stripped.split()) > _SMALL_TALK_MAX_WORDS:
        return ""
    return re.sub(r"\s+", " ", stripped).strip().lower()


def _looks_like_small_talk(question: str) -> bool:
    """No-LLM-call check: does the whole normalized message match any known
    small-talk phrase (canned or memory-based)? Used by
    `_correct_premature_terminal` to decide whether the planner's own
    "respond" choice should be trusted as-is."""
    normalized = _normalize_small_talk(question)
    return bool(normalized) and normalized in _SMALL_TALK_PHRASES


def _classify_fast_path(question: str) -> tuple[str, str] | None:
    """
    Zero-LLM-call classification for the pre-planner fast path (see
    `Agent._run_impl`/`_run_stream_impl`): returns `("canned", category)` if
    `question` is a pure greeting/thanks/farewell/ack/identity message (no
    Groq call needed at all — see `_CANNED_RESPONSES`), `("memory", "")` if
    it's a meta-conversation question needing real memory content, or
    `None` if it needs the normal planner loop. This is strictly a SUBSET
    of what `_looks_like_small_talk` matches — every fast-path hit is also
    small talk `_correct_premature_terminal` would trust, so this can never
    fast-path a message the planner would have routed to retrieval.
    """
    normalized = _normalize_small_talk(question)
    if not normalized:
        return None
    for category, phrases in _CANNED_PHRASES.items():
        if normalized in phrases:
            return ("canned", category)
    if normalized in _MEMORY_SMALL_TALK_PHRASES:
        return ("memory", "")
    return None


class Agent:
    def __init__(self, conversation_id: str = settings.DEFAULT_CONVERSATION_ID):
        self.conversation_id = conversation_id

        self.llm = AgentLLM()

        self.memory_manager = MemoryManager(
            llm=LLMTextGenerator(),
            conversation_id=conversation_id,
        )

        # The document this conversation is currently "about" — updated
        # whenever a retrieval step returns chunks from a document, or a
        # report is generated for one. Lets "اعمل تقرير" / "generate a
        # report" resolve which file to use without the user repeating
        # its name, as long as there's a document already in play.
        self.active_document: str | None = None

        self.tools = build_tools(
            memory_text_provider=self.memory_manager.as_prompt_text,
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

        fast_result = self._fast_path_answer(question, context)
        if fast_result is not None:
            self._remember(question, fast_result)
            return fast_result

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

        # Zero/one-call deterministic short-circuit for pure small talk —
        # see `_classify_fast_path` / `_fast_path_answer`'s docstring
        # (non-streaming counterpart). Skips the planner loop entirely.
        fast_classification = _classify_fast_path(question)

        if fast_classification is not None:
            kind, category = fast_classification
            context.observations.append({
                "tool": "respond", "status": "fast_path", "category": category or "memory",
            })
            if kind == "canned":
                full_text = _CANNED_RESPONSES[category]["ar" if detected_lang == "ar" else "en"]
                yield {"type": "token", "text": full_text}
            else:
                with timing.stage("memory_loading"):
                    memory_text = self.memory_manager.as_prompt_text()
                with timing.stage("llm_generation"):
                    for piece in rag_service.answer_from_memory_stream(question, memory_text, lang=detected_lang):
                        full_text += piece
                        yield {"type": "token", "text": piece}
            context.answer = full_text.strip()
        else:
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
                memory_text = self.memory_manager.as_prompt_text()
                with timing.stage("llm_generation"):
                    for piece in rag_service.generate_answer_stream(
                        question, context.documents, lang=context.language, memory=memory_text
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
        memory_text = self.memory_manager.as_prompt_text()

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

    def _fast_path_answer(self, question: str, context: ExecutionContext) -> ExecutionContext | None:
        """
        Zero/one-call deterministic short-circuit for pure small talk (see
        `_classify_fast_path`) — used by the non-streaming `_run_impl`
        before the planner loop starts. Returns a finished
        `ExecutionContext` (the planner/generation loop is skipped
        entirely) if `question` matches, else `None` so the caller falls
        through to the normal ReAct loop completely unchanged.

        A "canned" match costs ZERO Groq calls (a fixed localized reply).
        A "memory" match still needs one real generation call over actual
        conversation content ("what did I just ask?"), but skips the
        planner call that would otherwise precede it — the planner would
        deterministically be forced to "respond" here anyway (see
        `_correct_premature_terminal`), so skipping straight to it changes
        no behavior, only latency/token cost.
        """
        classification = _classify_fast_path(question)
        if classification is None:
            return None

        kind, category = classification
        context.observations.append({
            "tool": "respond", "status": "fast_path", "category": category or "memory",
        })

        if kind == "canned":
            context.answer = _CANNED_RESPONSES[category]["ar" if context.language == "ar" else "en"]
        else:
            with timing.stage("memory_loading"):
                memory_text = self.memory_manager.as_prompt_text()
            with timing.stage("llm_generation"):
                context.answer = rag_service.answer_from_memory(question, memory_text, lang=context.language)

        return context

    def _build_messages(self, question: str, context: ExecutionContext) -> list[dict]:
        with timing.stage("memory_loading"):
            memory_text = self.memory_manager.as_prompt_text() or "(none)"
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

        Only the "nothing retrieved yet" branch below is limited to firing
        once per turn (before any retrieval has happened). A SECOND,
        independent backstop runs first, on EVERY iteration: the planner
        can also discard already-retrieved evidence on a LATER step —
        after a prior "retrieve" this turn genuinely found relevant
        chunks, it can still (nondeterministically) choose "respond"
        (memory-only) instead of "generate" on the next step, throwing
        that evidence away. Confirmed via direct testing: retrieval
        succeeded (real chunk, cross-encoder score 0.28) yet the planner
        still answered "could you please rephrase your question?" from
        memory alone, with empty sources. The "nothing retrieved yet"
        check below never catches this — it returns the action as-is the
        moment context.documents is non-empty, which is exactly the state
        this second case needs corrected, not skipped.
        """
        if action.action not in (ToolName.RESPOND, ToolName.GENERATE):
            return action

        if action.action == ToolName.RESPOND and context.documents:
            # Deterministic, zero extra Groq calls (same technique as the
            # retrieve-forcing branch below) — and always safe: generate_
            # answer() already handles "documents present but don't
            # actually answer this" correctly (it reports "not available"
            # rather than hallucinating from memory — see the prompt's own
            # HARD RULE this mirrors), so forcing "generate" here can only
            # ever improve on discarding the retrieved evidence entirely.
            log.warning(
                f"Agent chose 'respond' with {len(context.documents)} document(s) already "
                f"retrieved this turn for question={question!r} — deterministically forcing "
                "'generate' so the retrieved evidence isn't discarded (no extra Groq call)."
            )
            return GenerateAction(
                thought=(
                    "Deterministic override: previous action discarded already-retrieved "
                    "documents by choosing 'respond' instead of 'generate'."
                ),
                action=ToolName.GENERATE,
                arguments=GenerateArguments(question=question[:500]),
            )

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
