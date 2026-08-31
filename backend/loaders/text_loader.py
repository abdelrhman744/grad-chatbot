"""Plain text/Markdown and JSON loading."""

import json
from typing import List

from langchain_core.documents import Document

from .base import make_meta


def load_text(filename: str, data: bytes) -> List[Document]:
    ext = filename.lower().rsplit(".", 1)[-1]
    file_type = "markdown" if ext == "md" else "txt"
    text = data.decode("utf-8", errors="replace").strip()
    return [Document(page_content=text, metadata=make_meta(filename, file_type))]


def load_json(filename: str, data: bytes) -> List[Document]:
    try:
        obj = json.loads(data.decode("utf-8", errors="replace"))
        text = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        text = data.decode("utf-8", errors="replace")
    return [Document(page_content=text, metadata=make_meta(filename, "json"))]
