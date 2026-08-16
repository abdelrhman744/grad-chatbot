"""
Task 5 — file type validation tests.

Covers: supported formats pass, unsupported extensions are rejected,
empty files are rejected, content that doesn't match its claimed
extension (malformed/renamed files) is rejected, and corrupt images are
caught via Pillow's own structural verification — all BEFORE anything
reaches the parser/OCR/RAG pipeline.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from utils.file_validation import (
    InvalidFileError,
    UnsupportedFileTypeError,
    validate_upload,
)


def _png_bytes(size=(20, 20), color=(255, 0, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class TestSupportedFilesPass:
    def test_valid_pdf_passes(self):
        data = b"%PDF-1.4\n%...rest of a pdf..."
        assert validate_upload("doc.pdf", data) == "pdf"

    def test_valid_png_passes(self):
        assert validate_upload("photo.png", _png_bytes()) == "image"

    def test_valid_txt_passes(self):
        assert validate_upload("notes.txt", "hello world مرحبا".encode("utf-8")) == "txt"

    def test_valid_docx_zip_signature_passes(self):
        # Real .docx files are ZIP archives — a bare ZIP local-file-header
        # signature is enough to pass this layer (deep OOXML structure
        # validation is the loader's job, not this cheap pre-filter's).
        data = b"PK\x03\x04" + b"\x00" * 40
        assert validate_upload("report.docx", data) == "docx"

    def test_valid_xlsx_zip_signature_passes(self):
        data = b"PK\x03\x04" + b"\x00" * 40
        assert validate_upload("sheet.xlsx", data) == "excel"

    def test_valid_legacy_xls_ole_signature_passes(self):
        data = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 40
        assert validate_upload("legacy.xls", data) == "excel"


class TestUnsupportedExtensionRejected:
    def test_exe_extension_rejected(self):
        with pytest.raises(UnsupportedFileTypeError):
            validate_upload("virus.exe", b"MZ\x90\x00")

    def test_no_extension_rejected(self):
        with pytest.raises(UnsupportedFileTypeError):
            validate_upload("noext", b"some content")

    def test_zip_extension_not_in_supported_set_rejected(self):
        with pytest.raises(UnsupportedFileTypeError):
            validate_upload("archive.zip", b"PK\x03\x04")


class TestEmptyFileRejected:
    def test_empty_bytes_rejected(self):
        with pytest.raises(InvalidFileError):
            validate_upload("empty.pdf", b"")


class TestMalformedFileRejected:
    def test_pdf_extension_with_plain_text_content_rejected(self):
        with pytest.raises(InvalidFileError):
            validate_upload("fake.pdf", b"this is just plain text, not a pdf")

    def test_docx_extension_with_plain_text_content_rejected(self):
        with pytest.raises(InvalidFileError):
            validate_upload("fake.docx", b"not a zip at all")

    def test_xls_extension_with_wrong_signature_rejected(self):
        with pytest.raises(InvalidFileError):
            validate_upload("fake.xls", b"definitely not an OLE file")

    def test_txt_extension_with_binary_null_bytes_rejected(self):
        with pytest.raises(InvalidFileError):
            validate_upload("fake.txt", b"\x00\x01\x02\x03binary garbage\x00\x00")

    def test_corrupt_truncated_png_rejected(self):
        # A real PNG signature but truncated/corrupt body — Pillow's
        # verify() must catch this, not just check the magic bytes.
        good = _png_bytes()
        truncated = good[: len(good) // 2]
        with pytest.raises(InvalidFileError):
            validate_upload("broken.png", truncated)

    def test_random_bytes_with_image_extension_rejected(self):
        with pytest.raises(InvalidFileError):
            validate_upload("fake.png", b"not an image at all, just text bytes")
