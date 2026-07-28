"""
short_memory.py

Holds the recent conversation messages in RAM for a single conversation.

This class does NOT touch disk, does NOT call the LLM, and does NOT
generate summaries — it only stores messages until MemoryManager decides
they need to be folded into the long-term summary.
"""

from __future__ import annotations


class ShortMemory:

    def __init__(self, max_messages: int = 25):

        self.max_messages = max_messages

        self.messages: list[dict] = []

    # =====================================================
    # Public Methods
    # =====================================================

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        return self.messages.copy()

    def count(self) -> int:
        return len(self.messages)

    def should_summarize(self) -> bool:
        return len(self.messages) > self.max_messages

    def clear(self) -> None:
        self.messages.clear()

    def is_empty(self) -> bool:
        return len(self.messages) == 0

    def keep_last(self, n: int) -> None:
        if n <= 0:
            self.clear()
        else:
            self.messages = self.messages[-n:]

    def __len__(self) -> int:
        return len(self.messages)
