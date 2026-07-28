"""
memory_manager.py

Coordinates:
- Short-term memory (recent raw messages, in RAM)
- Long-term memory (rolling conversation summary, persisted to disk)
- Automatic summarization when short-term memory grows too large

This is the memory system the Agent talks to. It never talks to Groq,
the vector store, or the retrieval pipeline directly — it only depends
on an injected LLM adapter exposing `.generate(prompt) -> str`, which
keeps memory fully reusable/testable outside the agent.
"""

from __future__ import annotations

import os

from .short_memory import ShortMemory
from .summarizer import Summarizer
from .summary_memory import SummaryMemory

# Hard backstop on summary length. The summarizer prompt *asks* for
# 6-8 sentences, but nothing stops the model from ignoring that on a
# given call — this caps the worst case so a runaway summary can never
# silently balloon every downstream prompt. Trims on a word boundary
# rather than mid-word/mid-sentence.
MAX_SUMMARY_CHARS = int(os.getenv("MEMORY_MAX_SUMMARY_CHARS", "700"))

# How much memory the *planner* sees vs. the terminal tools (generate/
# respond/compare). The planner only needs enough to route correctly
# (retrieve vs. respond vs. generate) — it doesn't need full recall,
# and it's re-invoked on every loop iteration (up to max_iterations),
# so trimming it here has a real multiplier on tokens saved. Terminal
# tools still get the full summary + window via as_prompt_text() with
# planner=False (the default), since answer quality actually depends
# on that.
PLANNER_WINDOW = int(os.getenv("MEMORY_PLANNER_WINDOW", "2"))
PLANNER_SUMMARY_CHARS = int(os.getenv("MEMORY_PLANNER_SUMMARY_CHARS", "200"))


def _truncate(text: str, max_chars: int) -> str:
    """Trim `text` to at most `max_chars` (INCLUDING the ellipsis marker),
    cutting on a word boundary instead of slicing mid-word/sentence."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    # Leave room for the ellipsis itself so the result never exceeds
    # max_chars overall.
    cut = text[: max_chars - 1].rsplit(" ", 1)[0].rstrip(",.;: ")
    return cut + "…"


class MemoryManager:

    def __init__(
        self,
        llm,
        conversation_id: str = "default",
        max_messages: int = 25,
        keep_recent: int = 6,
        window: int = 6,
        planner_window: int = PLANNER_WINDOW,
        planner_summary_chars: int = PLANNER_SUMMARY_CHARS,
    ):
        self.conversation_id = conversation_id
        self.keep_recent = keep_recent
        self.window = window
        self.planner_window = planner_window
        self.planner_summary_chars = planner_summary_chars

        self.short_memory = ShortMemory(max_messages)
        self.summary_memory = SummaryMemory()
        self.summarizer = Summarizer(llm_adapter=llm)

        # Load persisted summary once when the conversation starts.
        # Apply the hard cap here too, in case an older persisted
        # summary (written before MAX_SUMMARY_CHARS existed, or by a
        # smaller cap) is longer than the current limit.
        self.summary = _truncate(
            self.summary_memory.load_summary(self.conversation_id),
            MAX_SUMMARY_CHARS,
        )

    # =====================================================
    # Public Methods
    # =====================================================

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

    def get_last_assistant_message(self) -> str:
        """Most recent stored assistant reply, or "" if none yet. Used by
        RespondTool as a safety net to detect verbatim echoing of memory
        instead of a fresh answer to the current question."""
        for message in reversed(self.short_memory.get_messages()):
            if message["role"] == "assistant":
                return message["content"]
        return ""

    def as_prompt_text(self, planner: bool = False) -> str:
        """
        Render memory as plain text suitable for injecting into an LLM
        prompt: the long-term summary followed by the most recent
        messages within `window`.

        planner=True returns a much smaller view (short summary
        preview + last `planner_window` messages) for the routing
        model, which is re-invoked every loop iteration and only needs
        enough to decide the next action — not full recall. Terminal
        tools (generate/respond/compare) should keep planner=False so
        answer quality gets the full context.
        """
        window = self.planner_window if planner else self.window
        summary_chars = self.planner_summary_chars if planner else None

        parts = []

        if self.summary:
            summary_text = (
                _truncate(self.summary, summary_chars)
                if summary_chars is not None
                else self.summary
            )
            parts.append(f"Conversation summary so far:\n{summary_text}")

        recent = self.short_memory.get_messages()[-window:] if window else []
        if recent:
            formatted = "\n".join(
                f"{message['role'].capitalize()}: {message['content']}"
                for message in recent
            )
            parts.append(f"Recent messages:\n{formatted}")

        return "\n\n".join(parts)

    def reset(self) -> None:
        """Clear short-term memory and delete the persisted summary."""
        self.short_memory.clear()
        self.summary = ""
        self.summary_memory.delete_summary(self.conversation_id)

    # =====================================================
    # Internal
    # =====================================================

    def _summarize(self) -> None:
        """Fold short-term memory into the long-term summary and trim it."""
        messages = self.short_memory.get_messages()

        updated_summary = self.summarizer.summarize(
            old_summary=self.summary,
            new_messages=messages,
        )

        # Hard backstop: the summarizer prompt asks for 6-8 sentences,
        # but the model can ignore that on any given call. Cap here so
        # every downstream prompt (planner + terminal tools) has a
        # bounded worst case regardless.
        self.summary = _truncate(updated_summary, MAX_SUMMARY_CHARS)

        self.summary_memory.save_summary(self.conversation_id, self.summary)

        self.short_memory.keep_last(self.keep_recent)