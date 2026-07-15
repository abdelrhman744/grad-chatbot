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

from config import settings
from memory.llm_adapter import LLMTextGenerator
from memory.memory_manager import MemoryManager
from services.rag_service import detect_language

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

        self.tools = build_tools(memory_text_provider=self.memory_manager.as_prompt_text)

        self.max_iterations = settings.AGENT_MAX_ITERATIONS

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
