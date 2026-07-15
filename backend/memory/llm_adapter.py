"""
llm_adapter.py

The memory summarizer only needs a `.generate(prompt) -> str` interface.
This adapter wraps the single shared Groq-backed LLM instance from
services.rag_service so the memory package stays decoupled from any
specific LLM provider and could be pointed at a different model/provider
by swapping this one adapter.
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
            # Imported lazily to avoid a circular import at module load time
            # (rag_service is the owner of the shared LLM instance).
            from services.rag_service import get_llm
            self._llm = get_llm()
        return self._llm

    def generate(self, prompt: str) -> str:
        try:
            return str(self._get_llm().invoke(prompt)).strip()
        except Exception as e:
            log.error(f"LLMTextGenerator error: {e}")
            return ""
