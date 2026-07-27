"""
memory_manager.py

Coordinates:
- Short-term memory (recent raw messages, in RAM)
- Long-term memory (rolling conversation summary, persisted to disk)
- Automatic summarization when short-term memory grows too large

This is the memory system the Agent talks to. It never talks to the LLM
provider, Qdrant, or the RAG pipeline directly — it only depends on an injected
`.generate(prompt) -> str` LLM adapter, which keeps memory fully reusable
outside the agent context (e.g. for automated tests).
"""

from __future__ import annotations

from config import settings
from memory.short_memory import ShortMemory
from memory.summary_memory import SummaryMemory
from memory.summarizer import update_summary


class MemoryManager:
    def __init__(
        self,
        llm,
        conversation_id: str = settings.DEFAULT_CONVERSATION_ID,
        max_messages: int = settings.MEMORY_MAX_MESSAGES,
        keep_recent: int = settings.MEMORY_KEEP_RECENT,
    ):
        self.llm = llm
        self.conversation_id = conversation_id
        self.keep_recent = keep_recent

        self.short_memory = ShortMemory(max_messages)
        self.summary_memory = SummaryMemory()

        # Load persisted summary once when the conversation starts.
        self.summary = self.summary_memory.load_summary(self.conversation_id)

    # ── Public API ───────────────────────────────────────────────────────

    def add_message(self, role: str, content: str) -> None:
        """
        Add a single message to short-term memory. If the short-term
        buffer exceeds its limit, automatically fold it into the
        long-term summary and trim it back down.
        """
        self.short_memory.add_message(role, content)

        if self.short_memory.should_summarize():
            self._summarize()

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        """Add one complete User -> Assistant turn."""
        self.add_message("user", user_message)
        self.add_message("assistant", assistant_message)

    def get_recent_messages(self) -> list[dict]:
        return self.short_memory.get_messages()

    def get_summary(self) -> str:
        return self.summary

    def get_context(self) -> dict:
        """Full memory context, handed to the agent / generation prompts."""
        return {
            "summary": self.summary,
            "messages": self.get_recent_messages(),
        }

    def as_prompt_text(self, window: int = settings.MEMORY_WINDOW) -> str:
        """
        Render memory as plain text suitable for injecting into an LLM
        prompt: the long-term summary followed by the most recent
        `window` raw messages.
        """
        parts = []

        if self.summary:
            parts.append(f"Conversation summary so far:\n{self.summary}")

        recent = self.short_memory.get_messages()[-window:]
        if recent:
            formatted = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in recent
            )
            parts.append(f"Recent messages:\n{formatted}")

        return "\n\n".join(parts)

    def reset(self) -> None:
        """Clear short-term memory and delete the persisted summary."""
        self.short_memory.clear()
        self.summary = ""
        self.summary_memory.delete_summary(self.conversation_id)

    # ── Internal ─────────────────────────────────────────────────────────

    def _summarize(self) -> None:
        """Fold short-term memory into the long-term summary and trim it."""
        messages = self.short_memory.get_messages()

        self.summary = update_summary(
            llm=self.llm,
            old_summary=self.summary,
            messages=messages,
        )

        self.summary_memory.save_summary(self.conversation_id, self.summary)

        self.short_memory.keep_last(self.keep_recent)
