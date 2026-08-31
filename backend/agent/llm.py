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

from groq import BadRequestError
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
        # See config.py's AGENT_FALLBACK_MODEL docstring: used only for
        # Groq's own 400 json_validate_failed/json_generate_failed
        # (confirmed to happen occasionally even with the primary model at
        # temperature 0 — hosted inference is not perfectly deterministic).
        # Built lazily (not in __init__) since most turns never need it.
        self._fallback_llm = None

    def invoke(self, messages: List[dict], fallback_question: str = "") -> AgentAction:
        current_messages = list(messages)

        last_error: Exception | None = None
        used_fallback_model = False
        for attempt in range(self.max_retries + 1):
            try:
                llm = self._fallback_llm if used_fallback_model else self._llm
                raw = llm.chat(current_messages, json_mode=True)
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
            except BadRequestError as e:
                # Groq's OWN server-side JSON-mode validator rejected the
                # request outright — there is no model output to hand back
                # for self-correction (unlike the branch above), and unlike
                # a 429/network/auth failure (still re-raised unchanged
                # below) this is specific to this one generation attempt,
                # not a systemic account/infra problem — retrying the exact
                # same prompt against a DIFFERENT model is safe here, not
                # error-masking. One retry only; if the fallback model also
                # fails this way, that's treated the same as any other
                # malformed-output attempt (counts toward max_retries, then
                # falls through to the deterministic heuristic below).
                last_error = e
                if not used_fallback_model and self.model_name != settings.AGENT_FALLBACK_MODEL:
                    log.warning(
                        f"Agent LLM ({self.model_name}) rejected by Groq's JSON validator "
                        f"(attempt {attempt + 1}): {e} — retrying once against fallback "
                        f"model {settings.AGENT_FALLBACK_MODEL!r}."
                    )
                    if self._fallback_llm is None:
                        self._fallback_llm = get_agent_llm(model=settings.AGENT_FALLBACK_MODEL)
                    used_fallback_model = True
                else:
                    log.warning(f"Agent LLM JSON validation rejected (attempt {attempt + 1}): {e}")
            except Exception as e:
                # A real API/transport failure (rate limit, network, auth,
                # timeout, ...) is not something the heuristic fallback
                # below can meaningfully paper over: asking the model to
                # "return valid JSON" (the retry above) doesn't apply here,
                # and silently degrading to a fabricated "retrieve" action
                # previously produced a confusing, ungrounded final answer
                # with no sources -- while masking the real cause (e.g. a
                # Groq 429) from the user entirely. Let it propagate instead:
                # routes/chat.py and routes/ws.py already turn an uncaught
                # exception here into a clear, real-message error
                # response/event rather than a fake successful answer.
                log.error(f"Agent LLM invocation error (not swallowed — re-raising): {e}")
                raise

        log.warning(f"Falling back to heuristic action after {self.max_retries + 1} malformed-JSON attempts: {last_error}")
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
