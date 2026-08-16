"""
handwritten_ocr_service.py

Free, local, offline-capable handwritten OCR for Arabic and English, built
entirely on Hugging Face `transformers` (TrOCR) — no paid/external OCR API
of any kind. Completely separate from `services/ocr_service.py` (Tesseract),
which continues to handle printed-text OCR for scanned PDFs/images in the
existing upload pipeline unchanged.

Models (downloaded automatically by `transformers` on first use, cached
under the standard Hugging Face cache directory and reused on every
subsequent call/process restart — see config.py's HANDWRITTEN_OCR_* and
docker-compose.yml's `backend_model_cache` volume):
  - English: microsoft/trocr-base-handwritten
  - Arabic:  RayR1/trocr-base-arabic-handwritten

Design (mirrors services/embeddings_provider.py's pattern):
  - Each language's (processor, model) pair is loaded at most ONCE, lazily,
    on first request for that language, and cached in memory for the
    process lifetime — never reloaded per request.
  - A lock guards first-load per language so concurrent requests can't
    trigger duplicate downloads/loads.
  - CUDA is used automatically when available, CPU otherwise (mirrors
    utils/device.py's auto-detection, kept local here since this is an
    independent model family from the embedding/cross-encoder models that
    module resolves a device for).
  - Inference runs under `torch.inference_mode()`.

TrOCR handwritten checkpoints are trained on single text-line images, not
full document pages. A full-page image is handled by automatically
splitting it into individual line crops first (classic OpenCV ink-density
projection profile — no extra ML model, see `_segment_lines()`), running
TrOCR on each line, then joining the results — see `recognize()`.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from typing import List, Literal, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from config import settings

log = logging.getLogger("handwritten_ocr_service")

Language = Literal["ar", "en"]
SUPPORTED_LANGUAGES: Tuple[Language, ...] = ("ar", "en")

# Conservative preprocessing bounds. Aggressive thresholding/binarization is
# deliberately NOT applied — it tends to destroy Arabic diacritic dots and
# thin handwritten strokes, which hurts TrOCR more than it helps. Only
# orientation correction, RGB normalization, and mild up/downscaling +
# contrast are done.
_MIN_SIDE_PX = 64
_MAX_SIDE_PX = 2000


class InvalidImageError(ValueError):
    """Raised when the uploaded bytes are not a decodable image."""


class UnsupportedLanguageError(ValueError):
    """Raised when `language` is not one of SUPPORTED_LANGUAGES."""


class OCRModelError(RuntimeError):
    """Raised when the underlying model fails to load or run inference."""


def _preprocess(data: bytes) -> Image.Image:
    """
    Validate and lightly prepare an uploaded image for TrOCR.

    Steps: decode -> auto-orient via EXIF -> convert to RGB -> resize only
    if far outside a sane range (upscale tiny crops so thin strokes survive
    the model's internal resize, downscale huge images to bound memory/
    compute) -> mild contrast normalization. No binarization, no denoising,
    no rotation "correction" beyond EXIF — TrOCR's own image processor
    handles the final resize/normalization it needs.
    """
    if not data:
        raise InvalidImageError("Uploaded file is empty.")

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise InvalidImageError(f"File is not a valid image: {e}") from e

    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    w, h = img.size
    if min(w, h) == 0:
        raise InvalidImageError("Image has zero width or height.")

    short_side = min(w, h)
    long_side = max(w, h)

    if short_side < _MIN_SIDE_PX:
        scale = _MIN_SIDE_PX / short_side
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    elif long_side > _MAX_SIDE_PX:
        scale = _MAX_SIDE_PX / long_side
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)

    # Mild, non-destructive contrast normalization (stretches the existing
    # histogram instead of clipping/binarizing it) — helps faint pencil/pen
    # strokes without touching fine detail like Arabic dots.
    img = ImageOps.autocontrast(img, cutoff=1)

    return img


# ── Automatic line segmentation (classic CV, no extra ML model) ────────────
# TrOCR handwritten checkpoints expect single-line images; feeding a full
# page produces near-useless output (verified — see HANDWRITTEN_OCR.md).
# Rather than requiring the caller to pre-crop every line by hand, a
# page-sized image is automatically split into individual line crops here
# using a horizontal ink-density projection profile — a standard technique
# for document line segmentation, entirely classic computer vision (OpenCV
# + NumPy already used elsewhere in this project's OCR code, no new
# dependency, no ML model). It assumes roughly horizontal, non-overlapping
# lines (true for photographed/scanned notebook-style pages) and falls back
# to treating the whole image as a single line if it can't find a confident
# line structure — which also keeps the previous single-pre-cropped-line
# usage working unchanged (it just "detects" one line spanning the image).
#
# The "ink" mask used to build that profile is deliberately gradient-based
# (Scharr edge magnitude + Otsu), NOT a raw intensity/brightness threshold.
# Verified directly against real photographed notebook pages: plain
# intensity thresholding (both a fixed-fraction adaptiveThreshold mask and
# a straight global-Otsu mask) misclassified almost the entire page as
# "ink" because of ordinary photo artifacts — uneven lighting, a shadow, a
# laptop and desk visible in the background, paper texture — none of which
# is real handwriting. Edge/gradient magnitude is far less sensitive to
# smooth brightness variation (a shadow has almost no gradient) while
# still lighting up strongly wherever a pen stroke creates a sharp
# light/dark edge, which is what actually separated real text rows from
# background noise in that test.
#
# The row-classification threshold itself is computed PER IMAGE (1D Otsu
# over the smoothed row profile — same idea as 2D Otsu binarization, just
# applied to the 1D profile), not a single fixed constant: verified
# directly across 4 real photographed pages that a fixed fraction is not
# robust to how much a given photo's lighting/background noise happens to
# raise or lower the whole profile. `_LINE_OTSU_SCALE` nudges that
# per-image threshold up a bit — Otsu alone tends to sit slightly low here
# (splitting background-noise variation rather than isolating real ink),
# which under-splits adjacent lines into one merged band; 1.3x consistently
# produced clean, individually-sized bands across all 4 test pages.
_LINE_ROW_SMOOTH_PX = 9        # odd; moving-average window smoothing the row profile before classification — bridges the small ink/gap fluctuations within a single line of cursive handwriting into one coherent band instead of many tiny fragments
_LINE_OTSU_SCALE = 1.3         # multiplier applied to the per-image Otsu threshold on the row profile — see comment above
_LINE_ROW_MERGE_GAP_PX = 14    # merge row-bands separated by a (post-smoothing) gap <= this (keeps dots/diacritics attached to their line instead of splitting it)
_LINE_MIN_HEIGHT_PX = 14       # discard bands shorter than this (stray marks/ruling-line artifacts, not real text)
_LINE_MAX_HEIGHT_PX = 110      # force-split a run once it grows past this many rows without a real gap — caps the damage from a merged-lines failure (several capped-height chunks, still individually croppable) instead of one huge multi-line blob squashed into TrOCR's fixed input size; calibrated against real line heights (~15-100px) in the working (post-`_preprocess()`-resize) resolution
_LINE_MIN_INK_COL_FRAC = 0.015  # (within a band) a column counts as "text" once its edge-pixel fraction of the band height exceeds this
_LINE_VERTICAL_PAD_FRAC = 0.35  # extra padding above/below each detected band, as a fraction of the band's own height — covers ascenders/descenders/dots so characters aren't clipped
_LINE_HORIZONTAL_PAD_PX = 14    # extra left/right padding in pixels — avoids clipping leading/trailing strokes or punctuation
_LINE_MAX_COUNT = 80            # sanity cap: more "lines" than this on one page almost certainly means noisy segmentation, not real structure — fall back to whole image rather than run OCR 80+ times
_LINE_MAX_ASPECT_RATIO = 10.0   # width:height ceiling for a single cropped line — see the widen-if-too-wide step in _segment_lines(); 10:1 matches this module's own documented "good" ratio (see HANDWRITTEN_OCR.md's crop-aspect-ratio finding)


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Gradient-magnitude-based binary "ink" mask — see the module-level
    comment above `_segment_lines()` for why this is used instead of a
    plain intensity threshold. Returns a uint8 array of 0/255."""
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edge_bin = cv2.threshold(mag_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return edge_bin


def _smooth(profile: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(profile, kernel, mode="same")


def _otsu_1d(values: np.ndarray) -> float:
    """1D Otsu threshold over an arbitrary float profile (quantized to
    8-bit so cv2's Otsu implementation can be reused, then mapped back to
    the profile's own value range)."""
    vmin, vmax = float(values.min()), float(values.max())
    if vmax <= vmin:
        return vmin
    quant = ((values - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    thresh_q, _ = cv2.threshold(quant, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return vmin + (thresh_q / 255.0) * (vmax - vmin)


def _segment_lines(image: Image.Image) -> List[Image.Image]:
    """
    Split a page-ish image into a list of tightly-cropped single-line
    images, top to bottom. Falls back to `[image]` (the whole input, as one
    "line") if no confident line structure is found.

    Cropping is done against the ORIGINAL image passed in (already
    oriented/RGB/contrast-adjusted by `_preprocess()`, never binarized) —
    the ink-mask pass below exists ONLY to find where the handwriting is,
    it is never itself handed to TrOCR, so strokes/dots reach the model
    exactly as `_preprocess()` left them (no aggressive thresholding of the
    actual OCR input).
    """
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    ink = _ink_mask(gray)

    row_frac = _smooth((ink.sum(axis=1) / 255.0) / w, _LINE_ROW_SMOOTH_PX)
    row_thresh = _otsu_1d(row_frac) * _LINE_OTSU_SCALE
    row_is_text = row_frac > row_thresh

    bands: List[Tuple[int, int]] = []
    start: Optional[int] = None
    gap = 0
    for y in range(h):
        if row_is_text[y]:
            if start is None:
                start = y
            gap = 0
            if y - start >= _LINE_MAX_HEIGHT_PX:
                bands.append((start, y))
                start = y + 1
        elif start is not None:
            gap += 1
            if gap > _LINE_ROW_MERGE_GAP_PX:
                bands.append((start, y - gap))
                start = None
                gap = 0
    if start is not None:
        bands.append((start, h - 1))

    bands = [(y0, y1) for y0, y1 in bands if (y1 - y0) >= _LINE_MIN_HEIGHT_PX]

    if not bands or len(bands) > _LINE_MAX_COUNT:
        return [image]

    lines: List[Image.Image] = []
    for y0, y1 in bands:
        band_h = y1 - y0
        pad_v = max(4, int(band_h * _LINE_VERTICAL_PAD_FRAC))
        cy0 = max(0, y0 - pad_v)
        cy1 = min(h, y1 + pad_v)

        # Trim blank left/right margin within this band only, so a full
        # page-width band doesn't drag the huge blank margin most
        # handwritten lines leave on the page into every crop (this is the
        # single biggest lever on TrOCR accuracy for photographed
        # notebook-style pages — a very wide/thin crop gets squashed hard
        # by TrOCR's fixed square resize; see HANDWRITTEN_OCR.md).
        band_cols = ink[y0:y1 + 1, :].sum(axis=0) / 255.0 / band_h
        col_is_text = band_cols > _LINE_MIN_INK_COL_FRAC
        text_cols = np.where(col_is_text)[0]
        if len(text_cols) == 0:
            continue
        cx0 = max(0, int(text_cols[0]) - _LINE_HORIZONTAL_PAD_PX)
        cx1 = min(w, int(text_cols[-1]) + 1 + _LINE_HORIZONTAL_PAD_PX)

        # Bug fix (found via evaluate_ocr_preprocessing_check.py against
        # real IAM/KHATT handwriting samples): tightly cropping to just the
        # ink rows is exactly right for a photographed page's huge blank
        # margins, but on an input that was ALREADY a fairly tight single
        # line (e.g. a pre-cropped benchmark image, or a user-submitted
        # close-up crop), the same tight vertical crop can make the
        # resulting aspect ratio MORE extreme than the input — measured
        # directly turning a ~10:1 input into a ~20:1 crop — pushing it
        # further into the exact "very wide/thin crop gets squashed hard by
        # TrOCR's fixed square resize" failure this module's own docs
        # already warn about (see HANDWRITTEN_OCR.md). If the tight crop's
        # aspect ratio exceeds _LINE_MAX_ASPECT_RATIO, widen the crop
        # vertically (symmetrically, clamped to the image bounds) until it
        # no longer does — this keeps the original page-margin-trimming
        # benefit for genuinely wide page-width bands while no longer
        # actively hurting already-tight single-line input.
        crop_w = cx1 - cx0
        crop_h = cy1 - cy0
        if crop_h > 0 and crop_w / crop_h > _LINE_MAX_ASPECT_RATIO:
            target_h = crop_w / _LINE_MAX_ASPECT_RATIO
            extra = (target_h - crop_h) / 2
            cy0 = max(0, int(cy0 - extra))
            cy1 = min(h, int(cy1 + extra))

        lines.append(image.crop((cx0, cy0, cx1, cy1)))

    return lines if lines else [image]


class HandwrittenOCRService:
    """
    Lazily-loaded, in-memory-cached TrOCR models for Arabic and English
    handwriting recognition. One instance is shared for the process
    lifetime via `get_handwritten_ocr_service()`.
    """

    def __init__(self) -> None:
        self._device: Optional[str] = None
        self._lock = threading.Lock()
        # language -> (processor, model)
        self._models: dict = {}

    def _resolve_device(self) -> str:
        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info(f"Handwritten OCR will run on device={self._device!r}.")
        return self._device

    def _model_name_for(self, language: Language) -> str:
        return (
            settings.HANDWRITTEN_OCR_AR_MODEL
            if language == "ar"
            else settings.HANDWRITTEN_OCR_EN_MODEL
        )

    def _get_model(self, language: Language):
        """Return (processor, model) for `language`, loading it once
        (thread-safe, double-checked locking) on first use."""
        cached = self._models.get(language)
        if cached is not None:
            return cached

        with self._lock:
            cached = self._models.get(language)
            if cached is not None:
                return cached

            model_name = self._model_name_for(language)
            device = self._resolve_device()
            log.info(
                f"Loading handwritten OCR model for language={language!r}: "
                f"'{model_name}' on device={device!r} (one-time load; "
                "downloads from Hugging Face on first run, then reuses the "
                "local cache)..."
            )
            try:
                # Imported lazily so `transformers`/`torch` are only paid
                # for (import time, memory) if handwritten OCR is actually
                # used — every other model in this codebase (embeddings,
                # cross-encoder, Whisper) already follows this pattern.
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel

                processor = TrOCRProcessor.from_pretrained(model_name)
                model = VisionEncoderDecoderModel.from_pretrained(model_name)
                model.to(device)
                model.eval()
            except Exception as e:
                raise OCRModelError(
                    f"Could not load handwritten OCR model '{model_name}' "
                    f"for language={language!r}: {e}"
                ) from e

            log.info(f"Handwritten OCR model for language={language!r} loaded and ready.")
            self._models[language] = (processor, model)
            return processor, model

    def recognize(self, image_bytes: bytes, language: str) -> str:
        """
        Run handwritten OCR on an image and return the recognized text.

        Accepts either a single pre-cropped text line OR a full page: the
        image is automatically split into individual line crops first (see
        `_segment_lines()`), each line is OCR'd separately, and the results
        are joined with newlines in top-to-bottom order. An image that's
        already a single tight line degrades to exactly one detected
        "line" spanning the whole image, so existing single-line callers
        see unchanged behavior.

        Raises InvalidImageError / UnsupportedLanguageError for bad input,
        OCRModelError if the model can't be loaded or inference fails.
        """
        text, _debug = self.recognize_with_debug(image_bytes, language)
        return text

    def recognize_with_debug(self, image_bytes: bytes, language: str) -> Tuple[str, dict]:
        """
        Same as `recognize()`, but also returns a debug dict:
        `{"num_lines": int, "segment_time_s": float, "inference_time_s": float}`.
        Kept separate from `recognize()` so the public single-string return
        type callers already depend on (the API route, in particular)
        never changes shape — this is for internal instrumentation/testing.
        """
        if language not in SUPPORTED_LANGUAGES:
            raise UnsupportedLanguageError(
                f"Unsupported language {language!r} — must be one of {SUPPORTED_LANGUAGES}."
            )

        image = _preprocess(image_bytes)

        t0 = time.time()
        try:
            lines = _segment_lines(image)
        except Exception as e:
            log.warning(f"Line segmentation failed, falling back to whole image: {e}")
            lines = [image]
        t1 = time.time()

        texts = self._recognize_lines(lines, language)
        t2 = time.time()

        combined = "\n".join(t for t in (s.strip() for s in texts) if t)
        debug = {
            "num_lines": len(lines),
            "segment_time_s": t1 - t0,
            "inference_time_s": t2 - t1,
        }
        return combined, debug

    def _recognize_lines(self, images: List[Image.Image], language: str) -> List[str]:
        """
        Recognize a list of already-cropped line images. A single line
        (by far the most common call shape — one pre-cropped image, or a
        page that segmented into just one line) goes through
        `_recognize_image` directly; more than one line is batched through
        the model in RAM-bounded sub-batches of at most
        HANDWRITTEN_OCR_MAX_BATCH_SIZE instead of one model call per line.

        This used to be a plain `[self._recognize_image(img, language) for
        img in lines]` loop — strictly sequential, one full forward pass
        per line even for an 80-line page. Benchmarked before changing it
        (see scripts/evaluate_ocr_followup.py): batching measured a
        consistent ~15-54% latency reduction for English with IDENTICAL
        CER (not a quality/speed tradeoff) across two independent page
        sizes, and was latency-neutral (~same, occasionally a couple %
        slower) for Arabic — never a regression worth avoiding batching
        over. RSS increase was modest (tens of MB), bounded further here
        by the sub-batch size cap rather than batching an entire page's
        worth of lines (up to 80) in one call.

        Falls back to sequential per-image recognition for any sub-batch
        that raises (e.g. one malformed line image failing a batched
        forward pass) so one bad line can't take down an entire page.
        """
        if len(images) <= 1:
            return [self._recognize_image(img, language) for img in images]

        max_batch = max(1, settings.HANDWRITTEN_OCR_MAX_BATCH_SIZE)
        texts: List[str] = []
        for i in range(0, len(images), max_batch):
            batch = images[i:i + max_batch]
            try:
                texts.extend(self._recognize_batch(batch, language))
            except Exception as e:
                log.warning(
                    f"Batched OCR inference failed for a {len(batch)}-image sub-batch "
                    f"— falling back to sequential recognition for it: {e}"
                )
                texts.extend(self._recognize_image(img, language) for img in batch)
        return texts

    def _recognize_batch(self, images: List[Image.Image], language: str) -> List[str]:
        """Run TrOCR on several already-cropped line images in ONE batched
        forward pass. Raises OCRModelError on failure — the caller
        (`_recognize_lines`) decides whether to fall back to per-image
        recognition for this sub-batch."""
        processor, model = self._get_model(language)  # type: ignore[arg-type]
        device = self._resolve_device()

        try:
            import torch

            eos_token_id = model.config.eos_token_id
            if eos_token_id is None:
                eos_token_id = model.generation_config.eos_token_id
            pad_token_id = model.config.pad_token_id
            if pad_token_id is None:
                pad_token_id = model.generation_config.pad_token_id

            pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
            with torch.inference_mode():
                generated_ids = model.generate(
                    pixel_values,
                    max_new_tokens=settings.HANDWRITTEN_OCR_MAX_NEW_TOKENS,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    use_cache=True,
                )
            decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
        except Exception as e:
            raise OCRModelError(f"Batched handwritten OCR inference failed: {e}") from e

        return [t.strip() for t in decoded]

    def _recognize_image(self, image: Image.Image, language: str) -> str:
        """Run TrOCR on a single already-cropped PIL image (one line) and
        return its recognized text. Raises OCRModelError on failure."""
        processor, model = self._get_model(language)  # type: ignore[arg-type]
        device = self._resolve_device()

        try:
            import torch

            # Prefer the model's own base config's eos/pad token ids over the
            # checkpoint's generation_config.json, falling back to
            # generation_config's value only if the base config leaves it
            # unset (VisionEncoderDecoderModel's top-level config does for
            # microsoft/trocr-base-handwritten — verified directly). This
            # matters because at least one supported checkpoint
            # (RayR1/trocr-base-arabic-handwritten) ships a generation_config
            # with eos_token_id set equal to decoder_start_token_id — a
            # mismatch with its own model.config.eos_token_id (also verified
            # directly) that stops `generate()` from ever recognizing real
            # end-of-sequence, forcing it to run all the way to
            # max_new_tokens and pad the tail with [PAD] tokens instead of
            # stopping early. Harmless no-op for correctly configured
            # checkpoints.
            eos_token_id = model.config.eos_token_id
            if eos_token_id is None:
                eos_token_id = model.generation_config.eos_token_id
            pad_token_id = model.config.pad_token_id
            if pad_token_id is None:
                pad_token_id = model.generation_config.pad_token_id

            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            with torch.inference_mode():
                generated_ids = model.generate(
                    pixel_values,
                    max_new_tokens=settings.HANDWRITTEN_OCR_MAX_NEW_TOKENS,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    # Both supported checkpoints ship generation_config.json
                    # with use_cache=false (a training-time flag, not an
                    # inference requirement — VisionEncoderDecoderModel's
                    # decoder supports KV-caching fine at inference
                    # regardless). Left at that default, every generated
                    # token reprocesses the ENTIRE sequence so far instead of
                    # reusing cached key/value states — measured ~3-4x slower
                    # per call on CPU for identical output. Forcing it on
                    # here is a pure latency win with no effect on the
                    # generated text (verified: same output either way).
                    use_cache=True,
                )
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception as e:
            raise OCRModelError(f"Handwritten OCR inference failed: {e}") from e

        return text.strip()


_service: Optional[HandwrittenOCRService] = None
_service_lock = threading.Lock()


def get_handwritten_ocr_service() -> HandwrittenOCRService:
    """Return the shared, process-wide HandwrittenOCRService instance,
    creating it on first call only (no models are loaded here — that still
    happens lazily per-language inside `recognize()`)."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = HandwrittenOCRService()
    return _service
