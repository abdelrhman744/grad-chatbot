"""
generator.py

Generate responses using Llama 3 (Groq).

This is the single shared LLM client used by the chatbot layer
(generate/summarize/compare tools, respond tool, memory summarizer).
Centralizing it here means retry/error-handling only has to live in one
place instead of being duplicated in every tool.
"""

from __future__ import annotations

import os
import time
from typing import Iterator

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# A single client/module is reused by every Generator instance instead of
# opening a brand new Groq client per tool call.
_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


class Generator:

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
    ):
        self.client = _get_client()
        # Configurable via env so the model/limits can change per-deployment
        # without editing code (previously hardcoded).
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.temperature = (
            temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", 0.1))
        )
        # Capping max_tokens keeps latency and per-call cost predictable
        # instead of letting responses grow unbounded.
        self.max_tokens = max_tokens or int(os.getenv("LLM_MAX_TOKENS", 500))
        self.top_p = top_p if top_p is not None else float(os.getenv("LLM_TOP_P", 0.9))

    # =====================================================
    # Public Method
    # =====================================================

    def generate(
        self,
        prompt: str,
        retries: int = 2,
        fallback: str = "Sorry, I couldn't generate a response right now. Please try again.",
    ) -> str:
        """
        Call the LLM with a plain-text prompt and return the text answer.

        Retries transient failures (rate limits, timeouts, connection
        errors) with a short backoff instead of letting a single hiccup
        crash the whole conversation turn.
        """

        last_error: Exception | None = None

        for attempt in range(retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    messages=[{"role": "user", "content": prompt}],
                )

                content = response.choices[0].message.content
                return (content or "").strip()

            except Exception as error:
                last_error = error
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue

        # All retries exhausted — degrade gracefully instead of raising,
        # so a single LLM/API failure doesn't crash the agent loop.
        print(f"[Generator] generation failed after {retries + 1} attempts: {last_error}")
        return fallback

    def stream(
        self,
        prompt: str,
        retries: int = 2,
        fallback: str = "Sorry, I couldn't generate a response right now. Please try again.",
    ) -> Iterator[str]:
        """
        Same call as `generate`, but yields text deltas as they arrive
        instead of blocking for the full completion. Lets a caller (e.g.
        an HTTP endpoint) start sending output to the user immediately
        rather than waiting the whole generation time before the first
        byte, which is most of what "latency" means from the user's side.

        Retries a failed *connection attempt* the same way `generate`
        does; once tokens have started streaming, a mid-stream error is
        surfaced as a final fallback chunk instead of silently dropping
        the partial answer.
        """
        last_error: Exception | None = None

        for attempt in range(retries + 1):
            try:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                )

                yielded_any = False
                for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        yielded_any = True
                        yield delta

                if not yielded_any:
                    yield fallback
                return

            except Exception as error:
                last_error = error
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue

        print(f"[Generator] streaming failed after {retries + 1} attempts: {last_error}")
        yield fallback
