"""
summary_memory.py

Handles loading and saving long-term conversation summaries to local
JSON files. This is the persistence layer for the agent's long-term
memory, so a conversation's summary survives process restarts.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

DEFAULT_STORAGE_DIR = Path(__file__).parent / "storage"


class SummaryMemory:

    def __init__(self, storage_dir: str | Path | None = None):
        self.storage_dir = Path(storage_dir or DEFAULT_STORAGE_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # Public Methods
    # =====================================================

    def load_summary(self, conversation_id: str) -> str:
        file_path = self._get_file_path(conversation_id)

        if not file_path.exists():
            return ""

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            return data.get("summary", "")
        except Exception:
            return ""

    def save_summary(self, conversation_id: str, summary: str) -> None:
        file_path = self._get_file_path(conversation_id)

        data = {
            "conversation_id": conversation_id,
            "summary": summary,
            "updated_at": datetime.now().isoformat(),
        }

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def summary_exists(self, conversation_id: str) -> bool:
        return self._get_file_path(conversation_id).exists()

    def delete_summary(self, conversation_id: str) -> None:
        file_path = self._get_file_path(conversation_id)
        if file_path.exists():
            file_path.unlink()

    # =====================================================
    # Internal
    # =====================================================

    @staticmethod
    def _safe_id(conversation_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_\-]", "_", conversation_id) or "default"

    def _get_file_path(self, conversation_id: str) -> Path:
        return self.storage_dir / f"{self._safe_id(conversation_id)}.json"
