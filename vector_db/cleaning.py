"""
cleaning.py

Simple text cleaning for OCR output.
"""

import re


def Cleaner(text: str) -> str:
    """
    Clean OCR text while preserving its meaning.
    """

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Remove extra spaces
    text = re.sub(r"[ ]+", " ", text)

    # Remove extra blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Remove spaces before new lines
    text = re.sub(r" +\n", "\n", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


def is_meaningful(text: str) -> bool:
    """
    Reject chunks that are too short or contain no real script
    characters (Arabic or Latin) — e.g. OCR noise, stray punctuation,
    empty table cells. Keeps junk out of the vector store and the
    Whoosh index.
    """

    text = (text or "").strip()

    if len(text) < 15:
        return False

    return bool(re.search(r"[\u0600-\u06FFa-zA-Z]", text))