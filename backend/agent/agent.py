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
from collections import Counter

from config import settings
from memory.llm_adapter import LLMTextGenerator
from memory.memory_manager import MemoryManager
from services import rag_service
from services.rag_service import detect_language, build_sources_from_dicts

from .llm import AgentLLM
from .prompt import SYSTEM_PROMPT, USER_PROMPT
from .registry import build_tools
from .schemas import ExecutionContext, ToolName, TERMINAL_TOOLS

log = logging.getLogger("agent")


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
            active_document_provider=lambda: self.active_document,
            active_document_setter=self._set_active_document,
        )

        self.max_iterations = settings.AGENT_MAX_ITERATIONS

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
        detected_lang = detect_language(question) if language == "auto" else language
        context = ExecutionContext(language=detected_lang)

        for iteration in range(self.max_iterations):
            messages = self._build_messages(question, context)
            action = self.llm.invoke(messages, fallback_question=question)

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

                context = self._run_tool(action, context)
                context.retrieved_questions.append(action.arguments.question)
                self._update_active_document_from_retrieval(context)
                continue

            context = self._run_tool(action, context)

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
        detected_lang = detect_language(question) if language == "auto" else language
        context = ExecutionContext(language=detected_lang)
        full_text = ""

        for iteration in range(self.max_iterations):
            messages = self._build_messages(question, context)
            action = self.llm.invoke(messages, fallback_question=question)

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

                context = self._run_tool(action, context)
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
                    context = self._run_tool(action, context)
                    full_text = context.answer or ""
                else:
                    for piece in self._stream_terminal_action(action, context):
                        full_text += piece
                        yield {"type": "token", "text": piece}
                    self._finalize_stream(action, context, full_text)
                break

            context = self._run_tool(action, context)
        else:
            # Exhausted max_iterations — force a streamed final answer from
            # whatever context we have so the user always gets a response.
            log.warning(f"Agent hit max_iterations ({self.max_iterations}) without finishing.")
            memory_text = self.memory_manager.as_prompt_text()
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

    def _build_messages(self, question: str, context: ExecutionContext) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    question=question,
                    memory=self.memory_manager.as_prompt_text() or "(none)",
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

    def _run_tool(self, action, context: ExecutionContext) -> ExecutionContext:
        tool = self.tools[action.action.value]
        return tool.run(context=context, **action.arguments.model_dump())

    def _remember(self, question: str, context: ExecutionContext) -> None:
        answer = context.final_answer()
        if answer:
            self.memory_manager.add_turn(user_message=question, assistant_message=answer)

    def _debug_step(self, iteration: int, action) -> None:
        log.debug(f"--- iteration {iteration} ---")
        log.debug(f"thought: {action.thought}")
        log.debug(f"action: {action.action.value}")
        log.debug(f"arguments: {action.arguments.model_dump()}")
