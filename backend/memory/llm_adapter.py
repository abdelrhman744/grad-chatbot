"""
llm_adapter.py

The memory summarizer only needs a `.generate(prompt) -> str` interface.
This adapter wraps a shared Groq-backed LLM instance so the memory package
stays decoupled from any specific LLM provider and could be pointed at a
different model/provider by swapping this one adapter.
"""

from __future__ import annotations

import logging

log = logging.getLogger("memory.llm_adapter")


class LLMTextGenerator:
    """Minimal `.generate(prompt) -> str` wrapper around any LangChain-style LLM."""

    def __init__(self, llm=None):
        self._llm = llm

    def _get_llm(self):
        if self._llm is None:
            # Imported lazily to avoid a circular import at module load
            # time. Uses AGENT_MODEL (the small/fast planner model), not
            # GROQ_MODEL: fact extraction is structured JSON extraction
            # (discrete facts + an importance score), not open-ended
            # generation, and this call already runs off the
            # user-visible-latency critical path (see
            # memory_manager.py::_summarize_async's background thread) — so
            # the only thing this choice affects is which Groq rate-limit
            # pool it draws from, not turn latency. Keeping it off
            # GROQ_MODEL's pool leaves that pool's budget for the
            # user-visible final-answer generation call.
            from services.llm_provider import get_agent_llm
            self._llm = get_agent_llm()
        return self._llm

    def generate(self, prompt: str) -> str:
        try:
            return str(self._get_llm().invoke(prompt)).strip()
        except Exception as e:
            log.error(f"LLMTextGenerator error: {e}")
            return ""
