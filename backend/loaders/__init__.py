"""
loaders/

Per-file-type document loaders (PDF, DOCX, TXT/MD, JSON, images, Excel/CSV).
Each loader module exposes a `load(filename: str, data: bytes) -> list[Document]`
(or equivalently named) function with the same signature, so `registry.py`
can dispatch to any of them generically. `rag_service.py` only ever imports
from `loaders.registry` — it never talks to a specific loader module directly.
"""
