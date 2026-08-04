"""
memory_manager.py

Coordinates:
- Short-term memory (recent raw messages, in RAM)
- Long-term memory (a capped, deduplicated store of discrete facts,
  persisted to disk — see memory.summary_memory.FactStore)
- Automatic fact extraction when short-term memory grows too large

This is the memory system the Agent talks to. It never talks to the LLM
provider, Qdrant, or the RAG pipeline directly — it only depends on an injected
`.generate(prompt) -> str` LLM adapter, which keeps memory fully reusable
outside the agent context (e.g. for automated tests).
"""

from __future__ import annotations

import logging
import threading

from config import settings
from memory.fact_extractor import extract_facts
from memory.short_memory import ShortMemory
from memory.summary_memory import FactStore, SummaryMemory

log = logging.getLogger("memory_manager")


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

        self.short_memory = ShortMemory(max_messages, max_chars=settings.MEMORY_MAX_CHARS)
        self._persistence = SummaryMemory()

        # Load persisted facts once when the conversation starts.
        self.fact_store = FactStore(self._persistence.load_facts(self.conversation_id))

    # ── Public API ───────────────────────────────────────────────────────

    def add_message(self, role: str, content: str) -> None:
        """
        Add a single message to short-term memory. If the short-term
        buffer exceeds its limit (by message count or character budget),
        automatically fold it into the long-term fact store and trim it
        back down.
        """
        self.short_memory.add_message(role, content)

        if self.short_memory.should_summarize():
            self._summarize_async()

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        """Add one complete User -> Assistant turn."""
        self.add_message("user", user_message)
        self.add_message("assistant", assistant_message)

    def get_recent_messages(self) -> list[dict]:
        return self.short_memory.get_messages()

    def get_summary(self) -> str:
        """Long-term memory, rendered as text (kept for backward
        compatibility — this previously returned the free-text summary
        paragraph; it now returns the rendered fact list instead)."""
        return self.fact_store.render()

    def get_context(self) -> dict:
        """Full memory context, handed to the agent / generation prompts."""
        return {
            "summary": self.get_summary(),
            "messages": self.get_recent_messages(),
        }

    def as_prompt_text(self, window: int = settings.MEMORY_WINDOW) -> str:
        """
        Render memory as plain text suitable for injecting into an LLM
        prompt: the long-term fact list followed by the most recent
        `window` raw messages.
        """
        parts = []

        rendered_facts = self.fact_store.render()
        if rendered_facts:
            parts.append(f"Known facts from this conversation:\n{rendered_facts}")

        recent = self.short_memory.get_messages()[-window:]
        if recent:
            formatted = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in recent
            )
            parts.append(f"Recent messages:\n{formatted}")

        return "\n\n".join(parts)

    def reset(self) -> None:
        """Clear short-term memory and delete the persisted fact store."""
        self.short_memory.clear()
        self.fact_store = FactStore()
        self._persistence.delete_facts(self.conversation_id)

    # ── Internal ─────────────────────────────────────────────────────────

    def _summarize_async(self) -> None:
        """
        Fold short-term memory into the long-term fact store and trim it.

        Fact extraction is an LLM call, and saving is a disk write — doing
        both inline would add that latency to whichever turn happens to
        cross the summarization threshold (roughly 1 in MEMORY_MAX_MESSAGES
        turns). Since this only affects long-term memory for *future*
        turns, not the current turn's answer, snapshot what needs
        summarizing now (and trim the short-term buffer immediately, so it
        doesn't keep growing while the background thread runs), then do the
        actual extraction/merge/save on a background thread.
        """
        messages = self.short_memory.get_messages()
        existing_facts = list(self.fact_store.facts)
        self.short_memory.keep_last(self.keep_recent)

        def _run() -> None:
            try:
                update = extract_facts(
                    llm=self.llm,
                    existing_facts=existing_facts,
                    messages=messages,
                )
                self.fact_store.merge(update["facts"], update["remove"])
                self._persistence.save_facts(self.conversation_id, self.fact_store.facts)
            except Exception as e:
                log.warning(f"Background memory summarization failed: {e}")

        threading.Thread(target=_run, daemon=True).start()
