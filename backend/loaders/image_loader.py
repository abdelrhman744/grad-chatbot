"""Image loading via OCR."""

from typing import List

from langchain_core.documents import Document

from services.ocr_service import perform_ocr_image_bytes
from .base import make_meta, clean_text


def load(filename: str, data: bytes) -> List[Document]:
    text = perform_ocr_image_bytes(data)
    if not clean_text(text):
        return []
    return [Document(page_content=text, metadata={**make_meta(filename, "image"), "ocr": True})]
