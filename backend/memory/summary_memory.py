"""
summary_memory.py

Long-term memory as a capped, deduplicated FACT STORE, plus its disk
persistence — replacing the previous single free-text summary paragraph.

Why: a paragraph rewritten wholesale on every summarization cycle had no
programmatic guardrail against duplicate facts re-entering the text, no
importance ranking, and no size cap (only a soft "under 300 words" LLM
instruction) — so token usage could grow unbounded over a long conversation
and specific earlier facts could silently disappear on a later rewrite.
`FactStore` fixes this in code: near-duplicate facts are merged instead of
appended (`difflib.SequenceMatcher`), the total fact count is capped
(evicting lowest-importance/oldest first), and rendering is truncated to a
character budget regardless of how many facts have accumulated.

`SummaryMemory` is the disk persistence layer (one JSON file per
conversation under `settings.MEMORY_STORAGE_DIR`) and is backward
compatible with the old (version-1) `{"summary": "<free text>"}` file
format: an old file is transparently wrapped into a single legacy fact on
load, so memory_storage/*.json data written before this change isn't lost.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

from config import settings


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _similar(a: str, b: str, threshold: float = 0.92) -> bool:
    """
    Near-duplicate check for fact text. The threshold is deliberately high
    (0.92, not a looser value like 0.85): short factual sentences that
    differ in only one short token — e.g. "task: email Ahmed" vs "task:
    email Sara" — still score ~0.90-0.93 on character-level ratio despite
    being genuinely different facts, since almost every other character is
    identical. A high bar keeps true near-duplicates (the common real case:
    the LLM re-emitting an existing fact nearly verbatim on each
    summarization pass) merging correctly, while avoiding false-positive
    merges of distinct facts that happen to share a lot of boilerplate.
    """
    if not a or not b:
        return False
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio() >= threshold


class FactStore:
    """In-memory representation of one conversation's long-term facts, with
    dedup-aware merging and importance/recency-based capping and rendering."""

    def __init__(self, facts: list[dict] | None = None):
        self.facts: list[dict] = list(facts or [])

    def merge(self, new_facts: list[dict], remove_texts: list[str]) -> None:
        """
        Merge fact_extractor output into the store:
        - Facts whose text fuzzy-matches an entry in `remove_texts` are
          dropped (resolved tasks, superseded facts).
        - Each incoming fact that fuzzy-matches an existing fact REPLACES
          it (updated text/category/importance, fresh timestamp) instead
          of being appended as a duplicate; otherwise it's appended as new.
        - The store is then capped to settings.MEMORY_MAX_FACTS.
        """
        now = datetime.now().isoformat()

        for removed_text in remove_texts:
            self.facts = [f for f in self.facts if not _similar(f["text"], removed_text)]

        for new_fact in new_facts:
            entry = {**new_fact, "updated_at": now}
            match_idx = next(
                (i for i, f in enumerate(self.facts) if _similar(f["text"], entry["text"])),
                None,
            )
            if match_idx is not None:
                self.facts[match_idx] = entry
            else:
                self.facts.append(entry)

        self._cap()

    def _cap(self) -> None:
        max_facts = settings.MEMORY_MAX_FACTS
        if len(self.facts) <= max_facts:
            return
        # Rank ascending by (importance, recency) — the front of this list
        # is the lowest-importance, oldest facts, i.e. the eviction
        # candidates — without disturbing the store's own insertion order.
        ranked_for_eviction = sorted(
            self.facts, key=lambda f: (f.get("importance", 3), f.get("updated_at", ""))
        )
        evict_ids = {id(f) for f in ranked_for_eviction[: len(self.facts) - max_facts]}
        self.facts = [f for f in self.facts if id(f) not in evict_ids]

    def render(self, max_chars: int | None = None) -> str:
        """Render facts as a bullet list, most important/most recent first,
        truncated to a character budget so token usage stays bounded
        regardless of how many facts have accumulated over the conversation."""
        if not self.facts:
            return ""

        budget = settings.MEMORY_SUMMARY_MAX_CHARS if max_chars is None else max_chars

        # Stable sort twice: recency-descending first, then
        # importance-descending — ties within the same importance keep
        # their recency order thanks to sort stability.
        ranked = sorted(self.facts, key=lambda f: f.get("updated_at", ""), reverse=True)
        ranked = sorted(ranked, key=lambda f: f.get("importance", 3), reverse=True)

        lines: list[str] = []
        total = 0
        for f in ranked:
            line = f"- [{f.get('category', 'fact')}] {f['text']}"
            if lines and total + len(line) + 1 > budget:
                break
            lines.append(line)
            total += len(line) + 1

        return "\n".join(lines)


class SummaryMemory:
    """Disk persistence for one conversation's FactStore."""

    def __init__(self, storage_dir: str | None = None):
        self.storage_dir = Path(storage_dir or settings.MEMORY_STORAGE_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_id(conversation_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_\-]", "_", conversation_id) or "default"

    def _get_file_path(self, conversation_id: str) -> Path:
        return self.storage_dir / f"{self._safe_id(conversation_id)}.json"

    def load_facts(self, conversation_id: str) -> list[dict]:
        """Return the stored fact list, or [] if none exists yet. A legacy
        (pre-refactor) `{"summary": "..."}` file is wrapped into a single
        fact instead of being discarded."""
        file_path = self._get_file_path(conversation_id)

        if not file_path.exists():
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except Exception:
            return []

        if isinstance(data.get("facts"), list):
            return [f for f in data["facts"] if isinstance(f, dict) and f.get("text")]

        legacy_summary = data.get("summary")
        if legacy_summary and str(legacy_summary).strip():
            return [{
                "text": str(legacy_summary).strip(),
                "category": "fact",
                "importance": 3,
                "updated_at": data.get("updated_at") or datetime.now().isoformat(),
            }]

        return []

    def save_facts(self, conversation_id: str, facts: list[dict]) -> None:
        """Save or overwrite a conversation's fact store."""
        file_path = self._get_file_path(conversation_id)

        data = {
            "version": 2,
            "conversation_id": conversation_id,
            "facts": facts,
            "updated_at": datetime.now().isoformat(),
        }

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def facts_exist(self, conversation_id: str) -> bool:
        return self._get_file_path(conversation_id).exists()

    def delete_facts(self, conversation_id: str) -> None:
        file_path = self._get_file_path(conversation_id)
        if file_path.exists():
            file_path.unlink()
