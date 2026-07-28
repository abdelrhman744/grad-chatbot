"""
llm.py

Wrapper around the language model used by the ReAct agent.

Validates the model's JSON output against AgentAction. On a malformed
or invalid response, it retries with a correction message instead of
crashing the whole conversation turn, and falls back to a safe default
action if every attempt fails.
"""

from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv
from pydantic import TypeAdapter, ValidationError

from rag.generator import _get_client

from .schemas import AgentAction, GenerateAction, GenerateArguments, ToolName

logger = logging.getLogger(__name__)

load_dotenv()

MAX_ATTEMPTS = 3  # 1 initial attempt + 2 retries


class LLM:

    def __init__(self):

        # Reuses the same cached Groq client as Generator instead of opening
        # a second client per Agent instance.
        self.client = _get_client()

        # The planner only ever picks one of 5 fixed JSON actions — it
        # doesn't need 70B-class reasoning, and this call runs on every
        # single turn (sometimes twice), so its latency is on the
        # critical path far more than the final-answer generation is.
        # Default to a small/fast Groq model instead of inheriting the
        # 70B GROQ_MODEL used for actual answer generation. Still fully
        # overridable via AGENT_MODEL if a deployment needs the bigger
        # model's reasoning for planning (e.g. very ambiguous tool
        # choices).
        self.model = os.getenv("AGENT_MODEL", "llama-3.1-8b-instant")

        self.action_adapter = TypeAdapter(AgentAction)

    # =====================================================
    # Public Method
    # =====================================================

    def invoke(self, messages: list[dict], fallback_question: str = "") -> AgentAction:
        """
        Ask the model for the next action, validating and retrying on
        failure. Never raises for a bad model response — after
        MAX_ATTEMPTS it returns a safe default `generate` action so the
        agent loop can still produce an answer for the user.
        """

        working_messages = list(messages)
        last_error: str | None = None

        for attempt in range(MAX_ATTEMPTS):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=working_messages,
                )
            except Exception as error:
                last_error = f"LLM call failed: {error}"
                logger.warning("[Agent LLM] %s (attempt %s/%s)", last_error, attempt + 1, MAX_ATTEMPTS)
                continue

            content = response.choices[0].message.content or ""
            logger.debug("[Agent LLM] raw output: %s", content)

            try:
                data = json.loads(content)
                return self.action_adapter.validate_python(data)

            except json.JSONDecodeError:
                last_error = "The model returned invalid JSON."

            except ValidationError as error:
                last_error = f"The model returned an action that doesn't match the schema:\n{error}"

            logger.warning("[Agent LLM] %s (attempt %s/%s)", last_error, attempt + 1, MAX_ATTEMPTS)

            # Give the model one more shot with an explicit correction,
            # instead of failing the whole turn on the first bad response.
            working_messages = working_messages + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        f"Your previous response was invalid: {last_error}\n"
                        "Return ONLY valid JSON matching one of the action "
                        "shapes described in the system prompt. No markdown, "
                        "no explanation."
                    ),
                },
            ]

        # Every attempt failed — degrade gracefully instead of crashing
        # the conversation turn.
        logger.error("[Agent LLM] all %s attempts failed, using fallback action", MAX_ATTEMPTS)

        return GenerateAction(
            thought="Falling back to a direct answer after repeated invalid model output.",
            action=ToolName.GENERATE,
            arguments=GenerateArguments(question=fallback_question),
        )
