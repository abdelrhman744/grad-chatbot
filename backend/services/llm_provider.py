"""
llm_provider.py

Single shared LLM provider for the whole application, backed by the Groq
API (https://console.groq.com). This module is the ONLY place that talks
to Groq — every other module (rag_service, memory, agent) depends on the
small interface exposed here (`invoke`, `chat`) and never imports the
`groq` SDK directly.

Two access patterns are supported:

- `invoke(prompt)`      -> plain string completion, used by every
  single-prompt call in rag_service.py (translation, rephrasing, spelling
  correction, answer generation, summarization, comparison) and by the
  memory summarizer adapter. Mirrors the `.invoke()` interface the code
  previously used with `langchain_ollama.OllamaLLM`, so callers required
  no changes beyond the import.

- `chat(messages, json_mode=False)` -> plain string completion from a
  role-tagged message list, used by the agent's ReAct planning step
  (`agent/llm.py`), which needs strict JSON-object output.

Configuration is entirely environment-driven (see config.py):
  GROQ_API_KEY      - required, your Groq API key
  GROQ_MODEL        - main answer-generation model
  AGENT_MODEL       - model used for the agent's action-selection step
                       (defaults to GROQ_MODEL)
  LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TOP_P - generation parameters
"""

from __future__ import annotations

import logging
from typing import List, Optional

from groq import Groq

from config import settings

log = logging.getLogger("llm_provider")

_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not settings.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to backend/.env "
                "(see .env.example) or export it in your environment."
            )
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


class GroqLLM:
    """
    Thin wrapper around the Groq chat-completions API.

    Exposes `.invoke(prompt) -> str` so it is a drop-in replacement for the
    previous `langchain_ollama.OllamaLLM` instance used across the codebase.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        self.model = model or settings.GROQ_MODEL
        self.temperature = settings.LLM_TEMPERATURE if temperature is None else temperature
        self.max_tokens = settings.LLM_MAX_TOKENS if max_tokens is None else max_tokens
        self.top_p = settings.LLM_TOP_P if top_p is None else top_p

    # ── Single-prompt interface (used by rag_service.py / memory) ─────────
    def invoke(self, prompt: str) -> str:
        return self.chat([{"role": "user", "content": prompt}])

    # ── Chat-messages interface (used by agent/llm.py) ─────────────────────
    def chat(self, messages: List[dict], json_mode: bool = False) -> str:
        client = _get_client()

        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as e:
            log.error(f"Groq API call failed (model={self.model}): {e}")
            raise

        choice = response.choices[0]
        content = choice.message.content or ""
        return content.strip()


# ── Shared singleton instances ──────────────────────────────────────────────
# Mirrors the previous module-level `llm = OllamaLLM(...)` pattern so the
# rest of the codebase (rag_service.get_llm(), memory adapter, agent) can
# keep using a single shared instance without re-reading settings each call.

_shared_llm: Optional[GroqLLM] = None
_shared_agent_llm: Optional[GroqLLM] = None


def get_llm() -> GroqLLM:
    """Shared LLM instance used for answer generation, translation,
    summarization, comparison, and memory summarization."""
    global _shared_llm
    if _shared_llm is None:
        _shared_llm = GroqLLM(model=settings.GROQ_MODEL)
    return _shared_llm


def get_agent_llm(model: Optional[str] = None) -> GroqLLM:
    """Shared LLM instance used for the agent's action-selection step.
    Defaults to AGENT_MODEL (which itself defaults to GROQ_MODEL)."""
    global _shared_agent_llm
    if model is not None:
        return GroqLLM(model=model)
    if _shared_agent_llm is None:
        _shared_agent_llm = GroqLLM(model=settings.AGENT_MODEL)
    return _shared_agent_llm
