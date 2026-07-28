"""
agent.py

Main ReAct Agent. At each step it decides whether to retrieve documents,
generate an answer, summarize, compare, or respond directly from memory,
executes that single action, and loops until a terminal action is
reached or max_iterations is hit.

One Agent instance owns one conversation's memory (see agent/session.py
for the per-conversation registry used by callers that serve multiple
users/conversations).
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from rag.generator import Generator
from rag.memory.llm_adapter import LLMAdapter
from rag.memory.memory_manager import MemoryManager
from rag.utils.language import detect_language
from rag.utils.question_splitter import split_questions

from .llm import LLM
from .prompt import SYSTEM_PROMPT, USER_PROMPT
from .registry import build_tools
from .schemas import ExecutionContext, TERMINAL_TOOLS, ToolName

logger = logging.getLogger(__name__)

# Cap on how many sub-questions of a compound message are retrieved
# concurrently (see Agent._retrieve_compound_question). Bounded mainly
# to avoid one unusually long compound message from opening dozens of
# simultaneous connections to the vector store / BM25 index at once.
_MAX_COMPOUND_RETRIEVAL_WORKERS = int(os.getenv("AGENT_MAX_COMPOUND_RETRIEVAL_WORKERS", "5"))

# Keyword cues used only by the single-question fast path below (see
# `_guess_terminal_tool`). This is a cheap heuristic, not a classifier:
# it only has to distinguish the rare "compare"/"summarize" phrasing
# from the default "generate" case, so a false negative just costs one
# extra (unshortened) planner call rather than a wrong answer.
_COMPARE_KEYWORDS = ("compare", "vs.", " vs ", "versus", "قارن", "مقارنة", "الفرق بين")
_SUMMARIZE_KEYWORDS = ("summarize", "summary", "tl;dr", "لخص", "تلخيص", "ملخص")


def _guess_terminal_tool(question: str) -> str:
    """
    Best-effort guess of which terminal tool the planner would pick
    for a single (non-compound) question, used only to decide whether
    the fast path below is safe to take. Defaults to "generate", which
    is correct for the overwhelming majority of single-question turns.
    """
    lowered = question.lower()

    if any(keyword in lowered for keyword in _COMPARE_KEYWORDS):
        return ToolName.COMPARE.value

    if any(keyword in lowered for keyword in _SUMMARIZE_KEYWORDS):
        return ToolName.SUMMARIZE.value

    return ToolName.GENERATE.value


class Agent:

    def __init__(self, conversation_id: str = "default"):

        self.conversation_id = conversation_id

        self.llm = LLM()

        self.generator = Generator()

        self.memory_manager = MemoryManager(
            llm=LLMAdapter(self.generator),
            conversation_id=conversation_id,
        )

        self.tools = build_tools(
            memory_text_provider=self.memory_manager.as_prompt_text,
            last_assistant_provider=self.memory_manager.get_last_assistant_message,
        )

        # Configurable via env; defaults lowered from a hardcoded 10 to 6
        # (matching the sibling project) so a stuck planner can't rack up
        # an unbounded number of extra planning calls before the
        # max_iterations fallback forces a final answer.
        self.max_iterations = int(os.getenv("AGENT_MAX_ITERATIONS", "4"))

    # =====================================================
    # Public Methods
    # =====================================================

    def run(self, question: str, language: str = "auto", debug: bool = False) -> ExecutionContext:

        detected_language = detect_language(question) if language == "auto" else language

        context = ExecutionContext(question=question, language=detected_language)

        is_compound = self._retrieve_compound_question(question, context, debug=debug)

        # ---------------------------------------------
        # Guaranteed exit for compound questions
        # ---------------------------------------------
        # Every sub-question has already been retrieved deterministically
        # above, before the LLM-driven loop even starts. There's nothing
        # left for the planner to retrieve, so don't hand control to it
        # and hope it notices — a small/fast planner model can (and did)
        # keep re-issuing "retrieve" with slightly reworded questions
        # for a compound message, burning every iteration up to
        # max_iterations without ever reaching a terminal action. Go
        # straight to a terminal tool instead; this makes compound
        # turns terminate in one LLM call (the final-answer generation)
        # instead of up to `max_iterations` planner round-trips.
        if is_compound:
            terminal_tool = _guess_terminal_tool(question)
            context = self._run_terminal_tool(terminal_tool, context, question)
            self._remember(question, context)
            return context

        for iteration in range(self.max_iterations):

            messages = self._build_messages(question, context)

            action = self.llm.invoke(messages, fallback_question=question)

            if debug:
                self._debug_step(iteration + 1, action)

            # ---------------------------------------------
            # Prevent duplicate retrieval
            # ---------------------------------------------

            if action.action == ToolName.RETRIEVE:

                current_question = action.arguments.question.strip().lower()

                previous_questions = [q.lower() for q in context.retrieved_questions]

                if current_question in previous_questions:

                    context.observations.append(
                        {
                            "tool": "retrieve",
                            "question": action.arguments.question,
                            "status": "already_retrieved",
                        }
                    )

                    continue

                context = self._run_tool(action, context)

                context.retrieved_questions.append(action.arguments.question)

                # ---------------------------------------------
                # Single-question fast path
                # ---------------------------------------------
                # Compound questions never reach this loop at all (see
                # the guaranteed exit in run(), above), so by the time
                # we get here the message is always a single question.
                # This is the very first planner call (iteration 0), and
                # "retrieve -> generate" is by far the most likely next
                # step, so skip asking the planner a second time just to
                # confirm it and go straight to the terminal tool. This
                # removes one full LLM round-trip from the common
                # single-question turn.
                if iteration == 0:

                    terminal_tool = _guess_terminal_tool(question)

                    context = self._run_terminal_tool(terminal_tool, context, question)

                    break

                continue

            # ---------------------------------------------
            # Execute Tool
            # ---------------------------------------------

            context = self._run_tool(action, context)

            # ---------------------------------------------
            # Stop after a terminal action
            # ---------------------------------------------

            if action.action in TERMINAL_TOOLS:
                break

        else:
            # Exhausted max_iterations without reaching a terminal
            # action. Force a final answer from whatever context we
            # have so the user always gets a response instead of
            # silence.
            logger.warning("Agent hit max_iterations (%s) without finishing.", self.max_iterations)
            context = self.tools["generate"].run(context, question=question)

        self._remember(question, context)

        return context

    def reset_memory(self) -> None:
        self.memory_manager.reset()

    def run_stream(self, question: str, language: str = "auto", debug: bool = False):
        """
        Same planning loop as `run`, but for the final terminal step it
        yields the answer as it streams in from the model instead of
        blocking until the whole answer is ready. This is what lets an
        HTTP endpoint start sending bytes to the client immediately.

        Yields plain text chunks. The caller can recover the final
        ExecutionContext (for `.sources`, `.language`, etc.) from the
        `context` attribute set on this generator once it's exhausted,
        e.g.:

            gen = agent.run_stream(question)
            for chunk in gen:
                ...
            final_context = gen.context
        """
        return _StreamingRun(self, question, language, debug)

    # =====================================================
    # Deterministic Multi-Question Retrieval
    # =====================================================

    def _retrieve_compound_question(self, question: str, context: ExecutionContext, debug: bool = False) -> bool:
        """
        Detect and retrieve for each sub-question in a compound message
        BEFORE the LLM-driven loop starts.

        Splitting is a plain, repeatable heuristic (see
        rag.utils.question_splitter) — it always produces the same
        sub-questions for the same input, unlike asking the planner
        LLM to decide (turn by turn) whether and how to split a
        compound question, which is not fully reliable even at
        temperature=0.

        Single-question or non-question messages are unaffected:
        split_questions() returns them unchanged, so this is a no-op
        and the normal LLM-driven loop handles them exactly as before.

        Returns True if the message was actually compound (so the
        caller knows retrieval for every sub-question is already done
        and doesn't need to ask the planner to do it again).
        """

        sub_questions = split_questions(question)

        if len(sub_questions) < 2:
            return False

        retrieve_tool = self.tools["retrieve"]

        if debug:
            for sub_question in sub_questions:
                logger.debug("[Agent] pre-retrieving sub-question: %s", sub_question)

        # Each sub-question's retrieval (dense+sparse search, fusion,
        # rerank) is independent of the others, so fetch them all
        # concurrently instead of one after another — a 3-part question
        # used to take ~3x as long as a single retrieval. `fetch()` only
        # does the retrieval computation and touches no shared state, so
        # it's safe to run in parallel; `merge()` (which does mutate the
        # shared context) is still called sequentially afterward, in the
        # original sub-question order, to keep results deterministic.
        with ThreadPoolExecutor(
            max_workers=min(len(sub_questions), _MAX_COMPOUND_RETRIEVAL_WORKERS)
        ) as executor:
            fetch_results = list(
                executor.map(retrieve_tool.fetch, sub_questions)
            )

        for sub_question, (new_documents, error) in zip(sub_questions, fetch_results):
            context = retrieve_tool.merge(context, sub_question, new_documents, error)
            context.retrieved_questions.append(sub_question)

        return True

    # =====================================================
    # Build Messages
    # =====================================================

    def _build_messages(self, question: str, context: ExecutionContext) -> list[dict]:

        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    question=question,
                    memory=self.memory_manager.as_prompt_text() or "(none)",
                    retrieved_questions=", ".join(context.retrieved_questions) or "(none)",
                    observations=self._format_observations(context.observations),
                    documents=self._summarize_documents(context.documents),
                ),
            },
        ]

    @staticmethod
    def _format_observations(observations: list[dict]) -> str:
        if not observations:
            return "(none)"
        return "\n".join(
            json.dumps(observation, ensure_ascii=False) for observation in observations
        )

    # Cap on how many document snippets are shown to the planner (kept
    # small since this prompt is on the hot path — the planner just
    # needs enough signal to judge "do I have enough information yet",
    # not the full text every terminal tool will later see).
    _MAX_PLANNER_DOCUMENT_PREVIEWS = 5
    _PLANNER_DOCUMENT_SNIPPET_CHARS = 150

    @classmethod
    def _summarize_documents(cls, documents: list[dict]) -> str:
        """
        Short, human-readable preview of what's been retrieved so far,
        shown to the planner instead of a bare document count.

        A bare count ("Previously Retrieved Documents: 7") tells the
        planner nothing about whether those 7 chunks actually answer
        the question — it can only guess, which is one of the reasons
        a stuck planner keeps re-issuing "retrieve" instead of moving
        to a terminal action. Titles + short snippets let it actually
        judge relevance/coverage.
        """
        if not documents:
            return "(none)"

        lines = []
        for document in documents[: cls._MAX_PLANNER_DOCUMENT_PREVIEWS]:
            metadata = document.get("metadata", {})
            title = metadata.get("title", "Unknown")
            snippet = (document.get("text") or "").strip().replace("\n", " ")
            if len(snippet) > cls._PLANNER_DOCUMENT_SNIPPET_CHARS:
                snippet = snippet[: cls._PLANNER_DOCUMENT_SNIPPET_CHARS].rstrip() + "..."
            lines.append(f"- {title}: {snippet}")

        remaining = len(documents) - cls._MAX_PLANNER_DOCUMENT_PREVIEWS
        if remaining > 0:
            lines.append(f"... and {remaining} more chunk(s).")

        return "\n".join(lines)

    # =====================================================
    # Execute Tool
    # =====================================================

    def _run_tool(self, action, context: ExecutionContext) -> ExecutionContext:

        tool = self.tools[action.action.value]

        return tool.run(context=context, **action.arguments.model_dump())

    def _run_terminal_tool(self, tool_name: str, context: ExecutionContext, question: str) -> ExecutionContext:
        """
        Invoke a terminal tool by name outside of the normal planner
        flow (used by the single-question fast path). Handles the one
        signature difference among terminal tools: SummarizeTool.run()
        takes no `question` argument, unlike generate/compare/respond.
        """

        tool = self.tools[tool_name]

        if tool_name == ToolName.SUMMARIZE.value:
            return tool.run(context)

        return tool.run(context, question=question)

    # =====================================================
    # Save Memory
    # =====================================================

    def _remember(self, question: str, context: ExecutionContext) -> None:

        answer = context.final_text()

        if answer:
            self.memory_manager.add_turn(user_message=question, assistant_message=answer)

    # =====================================================
    # Debug
    # =====================================================

    def _debug_step(self, iteration: int, action) -> None:

        logger.debug("--- iteration %s ---", iteration)
        logger.debug("thought: %s", action.thought)
        logger.debug("action: %s", action.action.value)
        logger.debug("arguments: %s", action.arguments.model_dump())


class _StreamingRun:
    """
    Iterator returned by Agent.run_stream(). Mirrors Agent.run()'s
    planning loop exactly (same retrieve/dedup/terminal-action logic),
    except that once a terminal action is reached it streams that
    tool's output chunk-by-chunk (via the tool's `stream_run`, if it
    has one) instead of blocking for the full text.

    Kept as a separate class rather than folding into Agent.run()
    itself, so the existing non-streaming `run()` (used by chatbot.py
    and anything else that just wants a plain ExecutionContext) is
    untouched.
    """

    def __init__(self, agent: "Agent", question: str, language: str, debug: bool):
        self._agent = agent
        self._question = question
        self._language = language
        self._debug = debug
        self.context: ExecutionContext | None = None
        self._generator = self._run()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._generator)

    def _run(self):
        agent = self._agent
        question = self._question

        detected_language = (
            detect_language(question) if self._language == "auto" else self._language
        )
        context = ExecutionContext(question=question, language=detected_language)

        is_compound = agent._retrieve_compound_question(question, context, debug=self._debug)

        # Mirrors the non-streaming run()'s guaranteed exit for compound
        # questions — see the comment there. All sub-question retrieval
        # already happened above, so go straight to the (guessed)
        # terminal tool and stream it if possible, instead of letting
        # the planner loop potentially burn every iteration re-issuing
        # "retrieve".
        if is_compound:
            terminal_tool_name = _guess_terminal_tool(question)
            tool = agent.tools[terminal_tool_name]

            if terminal_tool_name == ToolName.SUMMARIZE.value:
                context = tool.run(context)
                text = context.final_text()
                if text:
                    yield text
            elif hasattr(tool, "stream_run"):
                yield from tool.stream_run(context=context, question=question)
            else:
                context = tool.run(context, question=question)
                text = context.final_text()
                if text:
                    yield text

            agent._remember(question, context)
            self.context = context
            return

        for iteration in range(agent.max_iterations):
            messages = agent._build_messages(question, context)
            action = agent.llm.invoke(messages, fallback_question=question)

            if self._debug:
                agent._debug_step(iteration + 1, action)

            if action.action == ToolName.RETRIEVE:
                current_question = action.arguments.question.strip().lower()
                previous_questions = [q.lower() for q in context.retrieved_questions]

                if current_question in previous_questions:
                    context.observations.append(
                        {
                            "tool": "retrieve",
                            "question": action.arguments.question,
                            "status": "already_retrieved",
                        }
                    )
                    continue

                context = agent._run_tool(action, context)
                context.retrieved_questions.append(action.arguments.question)

                # Single-question fast path — mirrors Agent.run() (see
                # the comment there for why this is safe). Compound
                # questions never reach this loop (handled above), so
                # skip the second planner round-trip and go straight to
                # the (guessed) terminal tool, streaming it if possible.
                if iteration == 0:

                    terminal_tool_name = _guess_terminal_tool(question)
                    tool = agent.tools[terminal_tool_name]

                    if terminal_tool_name == ToolName.SUMMARIZE.value:
                        context = tool.run(context)
                        text = context.final_text()
                        if text:
                            yield text
                    elif hasattr(tool, "stream_run"):
                        yield from tool.stream_run(context=context, question=question)
                    else:
                        context = tool.run(context, question=question)
                        text = context.final_text()
                        if text:
                            yield text

                    break

                continue

            # Terminal action reached — stream it if the tool supports
            # streaming, otherwise fall back to a single blocking call
            # and yield its full text as one chunk.
            tool = agent.tools[action.action.value]

            if hasattr(tool, "stream_run"):
                yield from tool.stream_run(context=context, **action.arguments.model_dump())
            else:
                context = tool.run(context=context, **action.arguments.model_dump())
                text = context.final_text()
                if text:
                    yield text

            break

        else:
            logger.warning("Agent hit max_iterations (%s) without finishing.", agent.max_iterations)

            tool = agent.tools["generate"]
            if hasattr(tool, "stream_run"):
                yield from tool.stream_run(context=context, question=question)
            else:
                context = tool.run(context, question=question)
                text = context.final_text()
                if text:
                    yield text

        agent._remember(question, context)
        self.context = context