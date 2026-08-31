"""
DOCX/DOC loading, with OCR support for embedded/scanned images.

A .docx file is itself a ZIP (OOXML) container; embedded pictures live
under `word/media/`. This module:
  1. Extracts native selectable text via Docx2txtLoader (unchanged from
     before — a normal text-based DOCX pays no OCR cost at all).
  2. Lists any embedded raster images directly via `zipfile` (no extra
     parsing dependency needed just for this) and OCR's them through the
     SAME services.ocr_service.perform_ocr_image_bytes used everywhere
     else, so Arabic + English OCR support and the OCR artifact cleanup in
     services/ocr_service.py both apply here automatically.
  3. Decides how to combine native text and OCR text:
       - no embedded images                               -> native text only
       - native text present AND embedded images present   -> native text +
         OCR text as an additional chunk (deduplicated against the native
         text so identical content isn't indexed twice)
       - next to no native text (essentially a scanned document made of
         images) -> OCR text becomes the document's content

Legacy .doc (binary OLE format, not a ZIP) still gets native text via
Docx2txtLoader; embedded-image OCR is skipped for it (zipfile can't parse
the format) — the same as before this change, not a regression.
"""

import io
import logging
import os
import re
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document

from config import settings
from services.ocr_service import perform_ocr_image_bytes
from .base import make_meta, clean_text

log = logging.getLogger("docx_loader")

_MEDIA_PREFIX = "word/media/"
# Raster formats Tesseract can OCR directly. .emf/.wmf (vector drawing
# objects — typically clipart/diagrams, not scanned content) are skipped.
_IMAGE_EXT_RE = re.compile(r"\.(png|jpe?g|bmp|tiff?|gif)$", re.IGNORECASE)


def _extract_embedded_images(data: bytes) -> List[bytes]:
    """Return the raw bytes of every embedded raster image in a DOCX file,
    sorted by filename (Word assigns image1.png, image2.png, ... in
    insertion order, so this approximates document order without a full
    relationship-graph walk)."""
    images: List[bytes] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = sorted(
                n for n in z.namelist()
                if n.startswith(_MEDIA_PREFIX) and _IMAGE_EXT_RE.search(n)
            )
            for name in names:
                try:
                    images.append(z.read(name))
                except Exception as e:
                    log.debug(f"Could not read embedded image {name}: {e}")
    except zipfile.BadZipFile:
        # Legacy .doc — not a ZIP container. No embedded-image OCR path
        # for this format; native text still comes from Docx2txtLoader.
        pass
    except Exception as e:
        log.warning(f"DOCX embedded image extraction failed: {e}")
    return images


def _ocr_embedded_images(images: List[bytes]) -> List[str]:
    """OCR every embedded image with bounded concurrency (same
    settings.OCR_MAX_CONCURRENT_PAGES knob perform_ocr_pdf_bytes uses for
    PDF pages), keeping only results that clear the same confidence bar
    used everywhere else in the ingestion pipeline (clean_text +
    OCR_MIN_TEXT_CHARS) — this is what filters out logos/decorative images,
    which typically OCR to empty or near-empty noise."""
    if not images:
        return []

    texts: List[Optional[str]] = [None] * len(images)

    def _run(index: int, img_bytes: bytes) -> None:
        try:
            text = perform_ocr_image_bytes(img_bytes)
            if clean_text(text) and len(text.strip()) >= settings.OCR_MIN_TEXT_CHARS:
                texts[index] = text
        except Exception as e:
            log.warning(f"DOCX embedded image {index} OCR failed, skipping: {e}")

    max_workers = max(1, min(settings.OCR_MAX_CONCURRENT_PAGES, len(images)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_run, i, img) for i, img in enumerate(images)]
        for f in futures:
            f.result()

    # Preserve document order; drop images that produced nothing usable.
    return [t for t in texts if t]


def _dedupe_against_native(ocr_texts: List[str], native_text: str) -> List[str]:
    """Drop OCR'd lines that are already present (near-verbatim) in the
    native text, so a docx where an image's caption/content is also typed
    out separately doesn't get indexed twice. Whole-image OCR blocks are
    kept as long as at least one of their lines is genuinely new — this
    only trims duplicate lines, never silently drops an entire image's
    contribution over one incidental overlapping line."""
    native_norm = re.sub(r"\s+", " ", native_text).lower()
    kept: List[str] = []
    for text in ocr_texts:
        unique_lines = []
        for line in text.splitlines():
            if not line.strip():
                continue
            norm = re.sub(r"\s+", " ", line.strip()).lower()
            if len(norm) >= 8 and norm in native_norm:
                continue  # already present in native text — skip the duplicate
            unique_lines.append(line)
        if unique_lines:
            kept.append("\n".join(unique_lines))
    return kept


def load(filename: str, data: bytes) -> List[Document]:
    ext = filename.lower().rsplit(".", 1)[-1]
    meta_base = make_meta(filename, ext)  # "docx" or "doc"

    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tf:
        tf.write(data)
        tmp_path = tf.name

    try:
        raw_docs = Docx2txtLoader(tmp_path).load()
    finally:
        os.unlink(tmp_path)

    for d in raw_docs:
        d.metadata = {**meta_base}
    raw_docs = [d for d in raw_docs if clean_text(d.page_content)]
    native_text = "\n".join(d.page_content for d in raw_docs)

    if not settings.ENABLE_PDF_OCR_FALLBACK:
        # Same master on/off switch PDF OCR fallback uses — DOCX OCR
        # follows it rather than introducing a second, separate knob.
        return raw_docs

    images = _extract_embedded_images(data)
    if not images:
        return raw_docs

    ocr_texts = _ocr_embedded_images(images)
    if not ocr_texts:
        return raw_docs

    if len(native_text.strip()) < settings.OCR_MIN_TEXT_CHARS:
        # Case 3: essentially a scanned document made of images — native
        # extraction found next to nothing, so the OCR'd image text IS the
        # document's content, not a supplement to it.
        log.info(
            f"DOCX '{filename}' looks scanned/image-only "
            f"({len(images)} embedded image(s)) — using OCR text as content"
        )
        return [
            Document(
                page_content="\n\n".join(
                    f"[Image {i + 1}]\n{t}" for i, t in enumerate(ocr_texts)
                ),
                metadata={**meta_base, "ocr_fallback": True, "image_count": len(ocr_texts)},
            )
        ]

    # Case 2: normal text-based DOCX that also has embedded images (e.g.
    # scanned appendix pages, screenshots with text) — keep native text as
    # the primary content and append de-duplicated OCR text as one more
    # chunk, so it still flows into the same chunking/embedding/Qdrant
    # pipeline as everything else.
    dedup_ocr_texts = _dedupe_against_native(ocr_texts, native_text)
    if not dedup_ocr_texts:
        return raw_docs

    log.info(
        f"DOCX '{filename}' has {len(images)} embedded image(s) with new OCR "
        f"text — merging with native text"
    )
    raw_docs.append(
        Document(
            page_content="\n\n".join(
                f"[Image {i + 1}]\n{t}" for i, t in enumerate(dedup_ocr_texts)
            ),
            metadata={**meta_base, "ocr_fallback": True, "image_count": len(dedup_ocr_texts)},
        )
    )
    return raw_docs
