"""
llm_adapter.py

Thin wrapper so the memory package doesn't depend directly on Groq /
the agent's LLM classes. It reuses rag.generator.Generator, the same
client the rest of the rag project already uses to talk to the model.
"""

from __future__ import annotations

from rag.generator import Generator


class LLMAdapter:

    def __init__(self, generator: Generator | None = None):
        self.generator = generator or Generator()

    def generate(self, prompt: str) -> str:
        return self.generator.generate(prompt)
