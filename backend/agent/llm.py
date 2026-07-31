"""
llm.py

Wraps the Groq-backed model used for the ReAct agent's action-selection
step. Uses Groq's JSON-object response mode and validates the result
against the AgentAction schema, retrying on malformed output before
falling back to a safe deterministic action (LLMs are not always
perfectly reliable at strict structured output, regardless of provider).
"""

from __future__ import annotations

import json
import logging
from typing import List

from pydantic import TypeAdapter, ValidationError

from config import settings
from services.llm_provider import get_agent_llm
from .schemas import (
    AgentAction,
    ToolName,
    RetrieveAction,
    RetrieveArguments,
)

log = logging.getLogger("agent.llm")


class AgentLLM:
    """Produces the agent's next structured action at each ReAct step."""

    def __init__(self, model: str | None = None, max_retries: int = 2):
        self.model_name = model or settings.AGENT_MODEL
        self._llm = get_agent_llm(model=self.model_name)
        self._adapter = TypeAdapter(AgentAction)
        self.max_retries = max_retries

    def invoke(self, messages: List[dict], fallback_question: str = "") -> AgentAction:
        current_messages = list(messages)

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self._llm.chat(current_messages, json_mode=True)
                data = json.loads(raw)
                return self._adapter.validate_python(data)
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                log.warning(f"Agent LLM produced invalid action (attempt {attempt + 1}): {e}")
                current_messages = current_messages + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was invalid JSON or did not match the "
                            "required schema. Return ONLY a single valid JSON object matching "
                            "one of the shapes described above."
                        ),
                    }
                ]
            except Exception as e:
                last_error = e
                log.error(f"Agent LLM invocation error: {e}")
                break

        log.warning(f"Falling back to heuristic action after planner failure: {last_error}")
        return self._fallback_action(fallback_question)

    # ── Internal ─────────────────────────────────────────────────────────

    def _fallback_action(self, question: str) -> AgentAction:
        """
        Deterministic fallback if the LLM repeatedly fails to produce a
        valid structured action. Defaults to retrieval so the loop makes
        forward progress instead of crashing the request.
        """
        return RetrieveAction(
            thought="Fallback: defaulting to retrieval after planner failure.",
            action=ToolName.RETRIEVE,
            arguments=RetrieveArguments(question=(question or "")[:500], top_k=5),
        )
