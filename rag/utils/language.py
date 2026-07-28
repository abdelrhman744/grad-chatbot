"""
language.py

Lightweight Arabic/English language detection used by the agent and its
tools to pick the right prompt template and refusal message. Not part of
the retrieval pipeline — this only affects how the chatbot talks to the
user.
"""

from __future__ import annotations

import re


def detect_language(text: str) -> str:
    """Return 'ar' if the text looks more Arabic than English, else 'en'."""
    text = text or ""

    ar_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    en_chars = len(re.findall(r"[a-zA-Z]", text))

    if ar_chars == 0 and en_chars == 0:
        return "en"

    return "ar" if ar_chars >= en_chars else "en"
