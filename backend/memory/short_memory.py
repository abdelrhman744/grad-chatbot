"""
short_memory.py

Holds the recent conversation messages in RAM for a single conversation.

This class does NOT touch disk, does NOT call the LLM, and does NOT
generate summaries — it only stores messages until MemoryManager decides
they need to be folded into the long-term summary.
"""

from __future__ import annotations


class ShortMemory:
    def __init__(self, max_messages: int = 25, max_chars: int | None = None):
        """
        Args:
            max_messages: Maximum number of messages before summarization
                is triggered by MemoryManager.
            max_chars: Optional total-character budget across all buffered
                messages. Summarization triggers on whichever limit is hit
                first, so a handful of very long messages can't silently
                balloon token usage before the message-count threshold fires.
        """
        self.max_messages = max_messages
        self.max_chars = max_chars
        self.messages: list[dict] = []

    def add_message(self, role: str, content: str) -> None:
        """Add a new message. role is 'user', 'assistant', or 'system'."""
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        return self.messages.copy()

    def count(self) -> int:
        return len(self.messages)

    def total_chars(self) -> int:
        return sum(len(m["content"]) for m in self.messages)

    def should_summarize(self) -> bool:
        if len(self.messages) > self.max_messages:
            return True
        if self.max_chars is not None and self.total_chars() > self.max_chars:
            return True
        return False

    def clear(self) -> None:
        self.messages.clear()

    def is_empty(self) -> bool:
        return len(self.messages) == 0

    def keep_last(self, n: int) -> None:
        """Keep only the last n messages (used after summarization)."""
        if n <= 0:
            self.clear()
        else:
            self.messages = self.messages[-n:]

    def get_last_message(self) -> dict | None:
        if self.is_empty():
            return None
        return self.messages[-1]

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        return f"ShortMemory(messages={len(self.messages)}, max_messages={self.max_messages})"
