"""
memory_manager.py

Conversational memory, short-term only: the recent raw messages of a
conversation, kept in RAM.

Long-term memory (LLM-based fact extraction + disk-persisted FactStore,
see the removed memory/fact_extractor.py and memory/summary_memory.py)
was removed. It added, per conversation, roughly every
MEMORY_MAX_MESSAGES messages:
  - an extra Groq call (fact extraction) on a background thread,
  - an uncapped `threading.Thread(...).start()` per event -- on a small
    box under concurrent load (many conversations crossing the threshold
    around the same time) this is unbounded thread/stack growth, not a
    fixed cost,
  - a non-atomic JSON file write per conversation (corruption risk on a
    crash/restart mid-write).
For a session-scoped assistant where conversations rarely outlive
agent/session.py's idle eviction window, that cost bought little: a new
Agent's ShortMemory starts empty on reload either way now, exactly as if
the conversation were new. Dropping it removes all three costs above.

This is the memory system the Agent talks to. It has no dependency on an
LLM, on disk, or on a background thread pool.
"""

from __future__ import annotations

from config import settings
from memory.short_memory import ShortMemory


class MemoryManager:
    def __init__(
        self,
        conversation_id: str = settings.DEFAULT_CONVERSATION_ID,
        max_messages: int = settings.MEMORY_MAX_MESSAGES,
        keep_recent: int = settings.MEMORY_KEEP_RECENT,
    ):
        self.conversation_id = conversation_id
        self.keep_recent = keep_recent

        self.short_memory = ShortMemory(max_messages, max_chars=settings.MEMORY_MAX_CHARS)

    # -- Public API ----------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        """
        Add a single message to short-term memory. If the buffer exceeds
        its limit (by message count or character budget), trim it back
        down to the most recent `keep_recent` messages -- purely in
        memory, no LLM call, no disk write.
        """
        self.short_memory.add_message(role, content)

        if self.short_memory.should_summarize():
            self.short_memory.keep_last(self.keep_recent)

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        """Add one complete User -> Assistant turn."""
        self.add_message("user", user_message)
        self.add_message("assistant", assistant_message)

    def get_recent_messages(self) -> list[dict]:
        return self.short_memory.get_messages()

    def get_context(self) -> dict:
        """Memory context, handed to the agent / generation prompts."""
        return {"messages": self.get_recent_messages()}

    def as_prompt_text(self, window: int = settings.MEMORY_WINDOW) -> str:
        """
        Render the most recent `window` raw messages as plain text
        suitable for injecting into an LLM prompt.
        """
        recent = self.short_memory.get_messages()[-window:]
        if not recent:
            return ""

        formatted = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in recent
        )
        return f"Recent messages:\n{formatted}"

    def reset(self) -> None:
        """Clear short-term memory."""
        self.short_memory.clear()
