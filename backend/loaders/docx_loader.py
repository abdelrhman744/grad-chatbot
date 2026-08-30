"""DOCX/DOC loading."""

import os
import tempfile
from typing import List

from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document

from .base import make_meta


def load(filename: str, data: bytes) -> List[Document]:
    ext = filename.lower().rsplit(".", 1)[-1]
    meta_base = make_meta(filename, ext)  # "docx" or "doc"

    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tf:
        tf.write(data)
        tmp_path = tf.name

    try:
        raw_docs = Docx2txtLoader(tmp_path).load()
        for d in raw_docs:
            d.metadata = {**meta_base}
        return raw_docs
    finally:
        os.unlink(tmp_path)
