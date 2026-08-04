"""
Single dispatch point mapping a file extension to: its file_type bucket,
its loader function, and its MinIO content-type. Adding support for a new
extension means adding one entry here (plus a loader module if it's a new
format) — nothing outside this file needs to change.
"""

from typing import Callable, List

from langchain_core.documents import Document

from . import pdf_loader, docx_loader, text_loader, image_loader, excel_loader

_EXT_TO_TYPE = {
    "pdf": "pdf",
    "docx": "docx", "doc": "doc",
    "txt": "txt", "md": "markdown",
    "json": "json",
    "jpg": "image", "jpeg": "image", "png": "image",
    "tiff": "image", "bmp": "image", "webp": "image",
    "xlsx": "excel", "xls": "excel", "csv": "excel",
}

_DISPATCH: "dict[str, Callable[[str, bytes], List[Document]]]" = {
    "pdf": pdf_loader.load,
    "docx": docx_loader.load, "doc": docx_loader.load,
    "txt": text_loader.load_text, "md": text_loader.load_text,
    "json": text_loader.load_json,
    "jpg": image_loader.load, "jpeg": image_loader.load, "png": image_loader.load,
    "tiff": image_loader.load, "bmp": image_loader.load, "webp": image_loader.load,
    "xlsx": excel_loader.load, "xls": excel_loader.load, "csv": excel_loader.load,
}

# Content-type by file_type bucket (fallback).
_CONTENT_TYPES_BY_TYPE = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc": "application/msword",
    "txt": "text/plain",
    "markdown": "text/markdown",
    "json": "application/json",
    "image": "application/octet-stream",
    "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

# Content-type overrides by extension, for cases where extensions in the
# same file_type bucket need different MIME types (legacy .xls, .csv).
_CONTENT_TYPES_BY_EXT = {
    "xls": "application/vnd.ms-excel",
    "csv": "text/csv",
}


def _ext(filename: str) -> str:
    return filename.lower().rsplit(".", 1)[-1] if "." in filename else ""


def get_file_type(filename: str) -> str:
    return _EXT_TO_TYPE.get(_ext(filename), "unknown")


def get_content_type(filename: str) -> str:
    ext = _ext(filename)
    if ext in _CONTENT_TYPES_BY_EXT:
        return _CONTENT_TYPES_BY_EXT[ext]
    return _CONTENT_TYPES_BY_TYPE.get(get_file_type(filename), "application/octet-stream")


def load_document_from_bytes(filename: str, data: bytes) -> List[Document]:
    loader_fn = _DISPATCH.get(_ext(filename))
    if loader_fn is None:
        return []
    return loader_fn(filename, data)
