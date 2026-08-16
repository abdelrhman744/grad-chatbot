"""PDF loading, with OCR fallback for scanned/image-only PDFs."""

import os
import tempfile
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from config import settings
from services.ocr_service import perform_ocr_pdf_bytes, perform_ocr_pdf_pages_bytes
from .base import make_meta, clean_text


def load(filename: str, data: bytes) -> List[Document]:
    meta_base = make_meta(filename, "pdf")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        tf.write(data)
        tmp_path = tf.name

    try:
        raw_docs = PyPDFLoader(tmp_path).load()
        text_body = "".join(d.page_content for d in raw_docs).strip()

        if not settings.ENABLE_PDF_OCR_FALLBACK:
            for i, d in enumerate(raw_docs):
                d.metadata = {**meta_base, "page": d.metadata.get("page", i)}
            return raw_docs

        if len(text_body) < settings.OCR_MIN_TEXT_CHARS:
            # Whole document has essentially no extractable text — it's
            # entirely scanned/image-only. OCR every page (unchanged from
            # the original behavior).
            ocr_text = perform_ocr_pdf_bytes(data)
            if clean_text(ocr_text):
                return [Document(page_content=ocr_text, metadata={**meta_base, "ocr_fallback": True})]
            return []

        for i, d in enumerate(raw_docs):
            d.metadata = {**meta_base, "page": d.metadata.get("page", i)}

        # Mixed document: the file as a WHOLE clears the text threshold
        # above (most pages have real extractable text), but any
        # INDIVIDUAL page with next-to-nothing extracted is very likely a
        # scanned/image page PyPDFLoader can't read. The old check only
        # looked at the whole-document concatenation, so these pages were
        # silently returned with their near-empty content and never OCR'd
        # — the surrounding pages' real text always kept the total above
        # the threshold. OCR only those specific weak pages (cheap: each
        # is rendered individually, not the whole PDF — see
        # perform_ocr_pdf_pages_bytes).
        weak_indices = [
            i for i, d in enumerate(raw_docs)
            if len((d.page_content or "").strip()) < settings.OCR_MIN_TEXT_CHARS
        ]
        if weak_indices:
            ocr_by_page = perform_ocr_pdf_pages_bytes(data, weak_indices)
            for i, ocr_text in ocr_by_page.items():
                if clean_text(ocr_text):
                    raw_docs[i].page_content = ocr_text
                    raw_docs[i].metadata["ocr_fallback"] = True

        return raw_docs
    finally:
        os.unlink(tmp_path)
