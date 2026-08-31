"""
Tests for loaders.docx_loader's DOCX OCR support (Requirement 2):
  - a normal text-based DOCX extracts native text and never triggers OCR
  - a DOCX with embedded images gets those images OCR'd and merged in
  - a DOCX that's essentially scanned images uses the OCR text as content
  - decorative/unreadable images contribute nothing (no OCR noise added)

Uses python-docx only to BUILD test fixtures (not a new runtime
dependency of the app itself) and monkeypatches
services.ocr_service.perform_ocr_image_bytes so these tests don't need a
real Tesseract binary / real scanned images.
"""

import io
import zipfile

import pytest

docx = pytest.importorskip("docx", reason="python-docx only needed to build test fixtures")

import loaders.docx_loader as docx_loader
from config import settings


def _make_docx(paragraphs, images=None) -> bytes:
    d = docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    buf = io.BytesIO()
    d.save(buf)
    data = buf.getvalue()

    if not images:
        return data

    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as zin, zipfile.ZipFile(out, "w") as zout:
        for item in zin.infolist():
            zout.writestr(item, zin.read(item.filename))
        for i, img_bytes in enumerate(images, start=1):
            zout.writestr(f"word/media/image{i}.png", img_bytes)
    return out.getvalue()


@pytest.fixture(autouse=True)
def _ocr_stub(monkeypatch):
    """Replace the real OCR call with a deterministic stub keyed off a
    marker prefix in the fake image bytes, and make sure OCR fallback is
    enabled / thresholds are predictable for these tests."""
    monkeypatch.setattr(settings, "ENABLE_PDF_OCR_FALLBACK", True)
    monkeypatch.setattr(settings, "OCR_MIN_TEXT_CHARS", 20)
    monkeypatch.setattr(settings, "OCR_MAX_CONCURRENT_PAGES", 4)

    def fake_ocr(data: bytes, strip_diacritics: bool = False) -> str:
        if data.startswith(b"ARABIC_SCAN"):
            return "هذا نص عربي تم استخراجه من صورة داخل مستند وورد"
        if data.startswith(b"ENGLISH_SCAN"):
            return "This is English text extracted from a scanned image inside a Word document"
        if data.startswith(b"LOGO"):
            return ""  # decorative image -> nothing usable
        return ""

    monkeypatch.setattr(docx_loader, "perform_ocr_image_bytes", fake_ocr)
    yield


def test_normal_text_docx_no_images_no_ocr(monkeypatch):
    calls = []
    monkeypatch.setattr(
        docx_loader, "perform_ocr_image_bytes",
        lambda data, strip_diacritics=False: calls.append(1) or "",
    )
    data = _make_docx([
        "This is a normal Word document.",
        "It has plenty of real text content here.",
    ])
    docs = docx_loader.load("normal.docx", data)

    assert len(docs) == 1
    assert "normal Word document" in docs[0].page_content
    assert not calls, "OCR must not run when there are no embedded images"


def test_mixed_docx_merges_native_and_ocr_text():
    data = _make_docx(
        ["This document has some real typed text.", "Plus an appendix scanned as an image."],
        images=[b"ARABIC_SCAN" + b"\x89PNG_fake_padding" * 5],
    )
    docs = docx_loader.load("mixed.docx", data)

    assert len(docs) == 2
    native_doc, ocr_doc = docs[0], docs[1]
    assert "real typed text" in native_doc.page_content
    assert "هذا نص عربي" in ocr_doc.page_content
    assert ocr_doc.metadata.get("ocr_fallback") is True


def test_scanned_only_docx_uses_ocr_text_as_content():
    data = _make_docx([""], images=[b"ENGLISH_SCAN" + b"\x89PNG_pad" * 10])
    docs = docx_loader.load("scanned.docx", data)

    assert len(docs) == 1
    assert "English text extracted from a scanned image" in docs[0].page_content
    assert docs[0].metadata.get("ocr_fallback") is True


def test_decorative_logo_image_adds_no_noise():
    data = _make_docx(
        ["Real text content that is definitely long enough to pass the threshold check."],
        images=[b"LOGO" + b"\x00" * 30],
    )
    docs = docx_loader.load("logo.docx", data)

    assert len(docs) == 1  # no second chunk added — logo OCR produced nothing usable
    assert "Real text content" in docs[0].page_content


def test_ocr_disabled_setting_skips_image_ocr(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_PDF_OCR_FALLBACK", False)
    calls = []
    monkeypatch.setattr(
        docx_loader, "perform_ocr_image_bytes",
        lambda data, strip_diacritics=False: calls.append(1) or "",
    )
    data = _make_docx(
        ["Some real text content here that is long enough."],
        images=[b"ARABIC_SCAN" + b"\x89PNG_pad" * 5],
    )
    docs = docx_loader.load("disabled.docx", data)

    assert not calls
    assert len(docs) == 1
