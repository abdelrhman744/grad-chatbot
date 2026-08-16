"""
file_validation.py

Task 5 — reject unsupported/malformed uploads BEFORE they reach the
parser/OCR/RAG pipeline, using extension + content (magic-byte / structural)
checks. Deliberately does NOT add a new dependency (no `python-magic`,
which needs a system libmagic that isn't installed anywhere this app
runs) — signature checks are plain byte-prefix comparisons, and images are
validated via Pillow (`pillow`, already a hard dependency of this project).

`loaders/registry.py::_EXT_TO_TYPE` remains the single source of truth for
which extensions this app claims to support at all; this module only adds
a content-sniffing layer on top of it.
"""

from __future__ import annotations

import io

from loaders.registry import SUPPORTED_EXTENSIONS, get_file_type
from PIL import Image, UnidentifiedImageError

_IMAGE_EXTENSIONS = frozenset({"jpg", "jpeg", "png", "tiff", "bmp", "webp"})
_TEXT_EXTENSIONS = frozenset({"txt", "md", "json", "csv"})

# Magic-byte signatures. PDF and the legacy OLE-compound Office formats
# (.doc/.xls) have one fixed signature each; the modern OOXML formats
# (.docx/.xlsx) are ZIP containers, so they share the ZIP local-file-header
# signature — this only proves "this is a real ZIP", not "this ZIP's
# internal XML is well-formed OOXML" (the loader itself will raise a clear
# error for that deeper case; this layer's job is to catch a renamed
# non-Office file cheaply, before any parsing is attempted).
_PDF_MAGIC = b"%PDF-"
_ZIP_MAGIC = b"PK\x03\x04"
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


class InvalidFileError(ValueError):
    """Raised when a file's content doesn't match what its extension claims
    (corrupt, truncated, or a renamed file of a different type)."""


class UnsupportedFileTypeError(ValueError):
    """Raised when a file's extension isn't one this app can process at all."""


def _ext(filename: str) -> str:
    return filename.lower().rsplit(".", 1)[-1] if "." in filename else ""


def validate_upload(filename: str, data: bytes) -> str:
    """
    Validate an uploaded file's extension and content before it's allowed
    into the parsing/OCR/chunking pipeline.

    Returns the validated file_type bucket (e.g. "pdf", "excel", "image")
    on success. Raises `UnsupportedFileTypeError` for an extension this app
    doesn't handle at all, or `InvalidFileError` for a supported extension
    whose content doesn't actually match (empty, corrupt, truncated, or a
    different file type renamed to a supported extension).
    """
    if not data:
        raise InvalidFileError(f"'{filename}' is empty.")

    ext = _ext(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '.{ext}' for '{filename}'. Supported types: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    if ext == "pdf":
        if not data.startswith(_PDF_MAGIC):
            raise InvalidFileError(
                f"'{filename}' has a .pdf extension but its content is not a valid PDF "
                "(missing the %PDF- signature)."
            )
    elif ext in ("docx", "xlsx"):
        if not data.startswith(_ZIP_MAGIC):
            raise InvalidFileError(
                f"'{filename}' has a .{ext} extension but its content is not a valid "
                "Office Open XML (ZIP-based) file."
            )
    elif ext in ("doc", "xls"):
        if not data.startswith(_OLE_MAGIC):
            raise InvalidFileError(
                f"'{filename}' has a .{ext} extension but its content is not a valid "
                "legacy Microsoft Office (OLE) file."
            )
    elif ext in _IMAGE_EXTENSIONS:
        try:
            img = Image.open(io.BytesIO(data))
            img.verify()
        except (UnidentifiedImageError, OSError, ValueError) as e:
            raise InvalidFileError(
                f"'{filename}' has an image extension but is not a valid, readable image: {e}"
            ) from e
    elif ext in _TEXT_EXTENSIONS:
        # A NUL byte in the first few KB is a strong, cheap signal of binary
        # content masquerading as text (real text files, Arabic/English
        # alike, never legitimately contain NUL).
        if b"\x00" in data[:4096]:
            raise InvalidFileError(
                f"'{filename}' has a .{ext} extension but appears to contain binary "
                "(non-text) content."
            )

    return get_file_type(filename)
