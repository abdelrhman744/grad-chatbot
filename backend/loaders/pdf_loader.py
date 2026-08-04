"""PDF loading, with OCR fallback for scanned/image-only PDFs."""

import os
import tempfile
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from config import settings
from services.ocr_service import perform_ocr_pdf_bytes
from .base import make_meta, clean_text


def load(filename: str, data: bytes) -> List[Document]:
    meta_base = make_meta(filename, "pdf")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        tf.write(data)
        tmp_path = tf.name

    try:
        raw_docs = PyPDFLoader(tmp_path).load()
        text_body = "".join(d.page_content for d in raw_docs).strip()

        if settings.ENABLE_PDF_OCR_FALLBACK and len(text_body) < 20:
            ocr_text = perform_ocr_pdf_bytes(data)
            if clean_text(ocr_text):
                return [Document(page_content=ocr_text, metadata={**meta_base, "ocr_fallback": True})]
            return []

        for i, d in enumerate(raw_docs):
            d.metadata = {**meta_base, "page": d.metadata.get("page", i)}
        return raw_docs
    finally:
        os.unlink(tmp_path)
