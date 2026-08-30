import logging
import re
import threading
import numpy as np
import cv2
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from config import settings

log = logging.getLogger("ocr_service")

pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD

# Full preprocessing-strategy x PSM-mode sweep, tried in this order.
# Element 0 of each list is the "most likely to work" combination and is
# always tried FIRST, alone — see _ocr_image_tiered. The remaining
# combinations only run if that first, cheap attempt looks unreliable
# (short output, or mostly non-alphanumeric noise), so a well-scanned page
# pays for exactly 1 Tesseract call instead of unconditionally paying for
# all of them.
OCR_STRATEGIES = ["adaptive", "otsu", "denoise", "sharpen", "contrast"]
OCR_PSM_MODES  = [6, 3, 11]

# Same idea, smaller sweep — used for PDF pages (perform_ocr_pdf_bytes /
# perform_ocr_pdf_pages_bytes), unchanged from the set already used there.
PDF_OCR_STRATEGIES = ["adaptive", "otsu", "denoise"]
PDF_OCR_PSM_MODES  = [6, 3]


# ── Preprocessing ──────────────────────────────────────────────────────────────

def _preprocess_for_ocr(img: np.ndarray, strategy: str) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    h, w = gray.shape
    if h < 800 or w < 600:
        scale = max(800 / h, 600 / w, 2.0)
        gray  = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if strategy == "adaptive":
        blurred   = cv2.GaussianBlur(gray, (5, 5), 0)
        processed = cv2.adaptiveThreshold(blurred, 255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    elif strategy == "otsu":
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, processed = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif strategy == "denoise":
        denoised = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)
        _, processed = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif strategy == "sharpen":
        blurred   = cv2.GaussianBlur(gray, (0, 0), 3)
        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        _, processed = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif strategy == "contrast":
        clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        _, processed = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        processed = gray

    return processed


def _run_tesseract(img: np.ndarray, psm: int = 6) -> str:
    config = f"--oem 1 --psm {psm} -l ara+eng"
    try:
        return pytesseract.image_to_string(Image.fromarray(img), config=config).strip()
    except Exception as e:
        log.debug(f"tesseract error (psm={psm}): {e}")
        return ""


def _merge_ocr_results(results: List[str]) -> str:
    if not results:
        return ""
    seen, lines = set(), []
    for result in sorted(results, key=len, reverse=True):
        for line in result.splitlines():
            key = re.sub(r"\s+", " ", line.strip().lower())
            if key and key not in seen:
                seen.add(key)
                lines.append(line.strip())
    return "\n".join(lines)


# ── Post-processing (RTL marks, fractured-line merge, digit handling) ──────
# Applied to every OCR result (image or PDF page) right before it's
# returned — see _postprocess_ocr_text, called from _ocr_image_tiered.
# Added after a real-sample investigation (see
# scripts/evaluate_printed_ocr_arabic.py) found these specific, reproducible
# failure patterns in Tesseract's raw `ara+eng` output.

# Bidi control characters (LRM U+200E, RLM U+200F) Tesseract sometimes
# emits verbatim around Arabic runs — invisible when rendered, but they
# break exact-text search/comparison downstream. Purely a storage/text
# concern; stripping them has no effect on how the text would render.
_RTL_MARKS_RE = re.compile(r"[‎‏]")

# Arabic diacritics (tashkeel/harakat) + tatweel. Opt-in only (see the
# strip_diacritics parameter on the public functions below) — most
# RAG/search use cases match on undiacritized text, but a caller that
# genuinely needs tashkeel preserved (e.g. a linguistics use case) must
# still be able to get the raw output.
_ARABIC_DIACRITICS_RE = re.compile(
    r"[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭـ]"
)

# Arabic-Indic digits (٠-٩) canonicalized to Western digits (0-9) so a
# document/query using either convention compares equal downstream. A
# lossless, order-preserving remap of digits Tesseract already recognized
# AS Arabic-Indic characters — separate from _reocr_digit_regions below,
# which recovers digits Tesseract misread as a different character/symbol
# entirely (verified directly: this happens far more often than a clean
# Arabic-Indic-vs-Western labeling difference — see the investigation).
_ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_WESTERN_DIGITS = "0123456789"
_DIGIT_NORMALIZE_TABLE = str.maketrans(_ARABIC_INDIC_DIGITS, _WESTERN_DIGITS)
_ALL_DIGIT_CHARS = set(_ARABIC_INDIC_DIGITS + _WESTERN_DIGITS)
_DIGIT_TOKEN_ALLOWED_EXTRA = set(".,/-:٫٬،")  # real number/date separators

_SHORT_LINE_MAX_LEN = 15   # a stripped line at/under this length is a merge candidate
_SHORT_LINE_MAX_RUN = 6    # cap: longer runs look like a real list/TOC, not a fracture
_LIST_MARKER_RE = re.compile(r"^[\-•*]|^\(?\d+[.)،]")


def strip_arabic_diacritics(text: str) -> str:
    """Remove Arabic diacritics (tashkeel/harakat) and tatweel from `text`.
    Never called implicitly — see the strip_diacritics parameter on
    perform_ocr_image_bytes / perform_ocr_pdf_bytes / extract_text."""
    return _ARABIC_DIACRITICS_RE.sub("", text or "")


def _normalize_digits(text: str) -> str:
    return text.translate(_DIGIT_NORMALIZE_TABLE)


def _is_digit_suspect_token(token: str) -> bool:
    """True for a token that's mostly digit characters but also contains
    something a real number/date wouldn't (garbled OCR noise mixed in with
    the digits — e.g. a stray letter or punctuation Tesseract emitted
    while misreading a digit run). A clean "2026" or "٢٩/٨/٢٠٢٦" is
    deliberately NOT flagged — nothing to fix. Deliberately conservative:
    verified directly (evaluate_printed_ocr_arabic.py) that some real
    garbled digit output (e.g. "51/84", "171040" — Tesseract's misreading
    of a 10-digit Arabic-Indic run) still passes this check uncaught,
    because it happens to look syntactically like a clean number/date —
    this heuristic catches noise-contaminated tokens, not tokens that are
    simply the wrong digits. See _reocr_digit_regions's docstring."""
    if not token:
        return False
    digit_count = sum(1 for c in token if c in _ALL_DIGIT_CHARS)
    if digit_count / len(token) < 0.4:
        return False
    extra = set(token) - _ALL_DIGIT_CHARS - _DIGIT_TOKEN_ALLOWED_EXTRA
    return bool(extra)


def _run_tesseract_digit_whitelist(img: np.ndarray) -> str:
    """Re-run Tesseract on the SAME image with the character set restricted
    to digits only. Verified directly (evaluate_printed_ocr_arabic.py):
    Tesseract's default ara+eng pass badly garbles an Arabic-Indic digit
    run surrounded by Arabic prose (e.g. "٠١٢٣٤٥٦٧٨٩" -> "١17740517085"),
    but a whitelist-only pass over the exact same image recovers it almost
    perfectly ("01234567894" — the correct 0-9 sequence, one stray extra
    digit). Never raises — returns "" on failure, same contract as
    _run_tesseract."""
    config = (
        "--oem 1 --psm 6 -l ara+eng -c tessedit_char_whitelist="
        + _WESTERN_DIGITS + _ARABIC_INDIC_DIGITS
    )
    try:
        return pytesseract.image_to_string(Image.fromarray(img), config=config).strip()
    except Exception as e:
        log.debug(f"digit-whitelist tesseract error: {e}")
        return ""


def _reocr_digit_regions(img: np.ndarray, text: str) -> str:
    """
    If `text` contains a digit-suspect token (see _is_digit_suspect_token),
    pay for ONE extra Tesseract call — a digit-only-whitelist pass over the
    same image (_run_tesseract_digit_whitelist) — and splice its
    cleanly-recognized digit runs back into `text` in left-to-right token
    order. Skipped entirely (no extra Tesseract call) when no token looks
    digit-suspect, which keeps the common non-numeric case at its existing
    single-call cost — same "only pay for what you need" design as
    _ocr_image_tiered's own tiering.

    Known limitation (found via re-running the fix against the same real
    garbled sample that motivated it — not assumed away): the trigger
    heuristic is intentionally conservative to avoid corrupting a
    legitimately-recognized number elsewhere in a document, so it does NOT
    fire on every garbled digit run — e.g. "51/84"/"171040" (Tesseract's
    actual misreading of a 10-digit Arabic-Indic string) both look
    syntactically like a clean number/date and pass _is_digit_suspect_token
    uncaught. The whitelist-only re-OCR pass itself IS accurate when it
    does run (verified directly: recovered "01234567894" — the correct
    0-9 sequence, one stray extra digit — from an image whose default pass
    produced "17740517085"); the gap is in reliably detecting WHEN to
    trigger it from text alone, not in the recovery itself.
    """
    tokens = text.split(" ")
    suspect_indices = [i for i, t in enumerate(tokens) if _is_digit_suspect_token(t)]
    if not suspect_indices:
        return text

    whitelist_text = _run_tesseract_digit_whitelist(img)
    digit_runs = re.findall(r"[0-9٠-٩]+", whitelist_text)
    if not digit_runs:
        return text

    for position, idx in enumerate(suspect_indices):
        if position < len(digit_runs):
            tokens[idx] = digit_runs[position]
    return " ".join(tokens)


def _merge_fractured_lines(text: str) -> str:
    """
    Merge runs of consecutive very-short lines that are almost certainly a
    single source line Tesseract's PSM mis-split into fragments — verified
    directly against a real ligature-heavy Arabic line
    (evaluate_printed_ocr_arabic.py): one input line came back as 6+
    separate short/garbled lines. Deliberately conservative: only lines
    with no blank-line separator between them, that don't look like an
    intentional list/heading marker (_LIST_MARKER_RE), and only up to
    _SHORT_LINE_MAX_RUN in a row — a longer run of short lines looks more
    like a genuine list/table-of-contents than a fracture, so it's left
    alone. Never touches "[Page N]" markers (perform_ocr_pdf_bytes) since
    those are always separated from surrounding text by blank lines.
    """
    lines = text.split("\n")
    merged: List[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped or len(stripped) > _SHORT_LINE_MAX_LEN or _LIST_MARKER_RE.match(stripped):
            merged.append(lines[i])
            i += 1
            continue

        run = [stripped]
        j = i + 1
        while (
            j < len(lines)
            and len(run) < _SHORT_LINE_MAX_RUN
            and lines[j].strip()
            and len(lines[j].strip()) <= _SHORT_LINE_MAX_LEN
            and not _LIST_MARKER_RE.match(lines[j].strip())
        ):
            run.append(lines[j].strip())
            j += 1

        merged.append(" ".join(run) if len(run) >= 2 else lines[i])
        i = j

    return "\n".join(merged)


def _postprocess_ocr_text(text: str, img: Optional[np.ndarray] = None) -> str:
    """Applied to every OCR result before it's returned. `img` (the
    preprocessed array OCR actually ran against), when available, enables
    the digit-region re-OCR pass; text-only cleanup still runs without it
    (e.g. the raw-fallback path in perform_ocr_image_bytes)."""
    if not text:
        return text
    # RTL marks stripped FIRST — a stray LRM/RLM glued directly onto a
    # digit token (observed directly: '‎٠‏') would otherwise
    # dilute its digit-character ratio enough to dodge
    # _is_digit_suspect_token's threshold.
    text = _RTL_MARKS_RE.sub("", text)
    if img is not None:
        try:
            text = _reocr_digit_regions(img, text)
        except Exception as e:
            log.debug(f"digit-region re-OCR post-process step failed, skipping: {e}")
    text = _merge_fractured_lines(text)
    text = _normalize_digits(text)
    return text


def _ocr_result_confident(text: str) -> bool:
    """
    Cheap, deterministic heuristic used ONLY to decide whether a single
    OCR attempt is good enough to skip the rest of the preprocessing x PSM
    sweep — NOT a claim of ground-truth accuracy, and not itself an OCR
    engine. Two checks, both on raw output already produced by Tesseract:
    long enough to plausibly be real content, and a large-enough fraction
    of non-whitespace characters are alphanumeric (Arabic or Latin letters,
    digits) rather than the sparse/garbled symbol noise a bad preprocessing
    choice typically produces. Works the same way for Arabic and English
    since `str.isalnum()` is Unicode-aware.
    """
    text = (text or "").strip()
    if len(text) < settings.OCR_MIN_TEXT_CHARS:
        return False
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return False
    alnum_ratio = sum(1 for c in non_ws if c.isalnum()) / len(non_ws)
    return alnum_ratio >= settings.OCR_MIN_ALNUM_RATIO


def _ocr_image_tiered(img: np.ndarray, strategies: List[str], psm_modes: List[int]) -> str:
    """
    Try the single most-likely-to-work (strategy, psm) combination first —
    element 0 of each list. If that result already looks confident (see
    `_ocr_result_confident`), return it immediately: this is the common
    case for a normal, reasonably clean scan, and it means paying for
    exactly ONE Tesseract invocation instead of unconditionally running
    every combination.

    Only if that first attempt looks weak/empty does this fall back to the
    full sweep (every remaining combination, merged exactly as before) —
    so a genuinely hard page still gets the same accuracy ceiling as
    before this change; nothing is removed, only reordered and gated.
    """
    best_strategy, best_psm = strategies[0], psm_modes[0]
    first_text = ""
    processed = None
    try:
        processed = _preprocess_for_ocr(img, best_strategy)
        first_text = _run_tesseract(processed, best_psm)
    except Exception as e:
        log.debug(f"tiered OCR first pass failed (strategy={best_strategy}, psm={best_psm}): {e}")

    if _ocr_result_confident(first_text):
        return _postprocess_ocr_text(first_text, processed)

    # Escalate: run every other combination too and merge everything,
    # exactly like the original always-run-all behavior — this only fires
    # for pages/images that actually need it.
    results = [first_text] if first_text else []
    for strategy in strategies:
        for psm in psm_modes:
            if strategy == best_strategy and psm == best_psm:
                continue
            try:
                processed_variant = _preprocess_for_ocr(img, strategy)
                text = _run_tesseract(processed_variant, psm)
                if text:
                    results.append(text)
            except Exception:
                continue
    merged = _merge_ocr_results(results)
    # Digit re-OCR needs ONE representative preprocessed image — the
    # first/best strategy's, same one used for the fast-path case above —
    # rather than re-running it against every escalation variant.
    return _postprocess_ocr_text(merged, processed)


def _decode_image_bytes(data: bytes) -> Optional[np.ndarray]:
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        from PIL import Image as PILImage
        import io
        pil = PILImage.open(io.BytesIO(data)).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


# ── Public API ─────────────────────────────────────────────────────────────────

def perform_ocr_image_bytes(data: bytes, strip_diacritics: bool = False) -> str:
    """Run OCR on raw image bytes (tiered — see _ocr_image_tiered).
    strip_diacritics=True additionally removes Arabic tashkeel/harakat from
    the result (see strip_arabic_diacritics) — off by default, since not
    every caller wants diacritics discarded."""
    merged = ""
    try:
        img = _decode_image_bytes(data)
        if img is not None:
            merged = _ocr_image_tiered(img, OCR_STRATEGIES, OCR_PSM_MODES)
    except Exception as e:
        log.error(f"OCR image error: {e}")

    if not merged:
        try:
            import io
            merged = pytesseract.image_to_string(
                Image.open(io.BytesIO(data)), lang="ara+eng", config="--oem 1 --psm 6"
            ).strip()
            merged = _postprocess_ocr_text(merged)
        except Exception:
            pass

    if strip_diacritics:
        merged = strip_arabic_diacritics(merged)

    log.info(f"OCR image → {len(merged)} chars")
    return merged


def perform_ocr_image_path(file_path: str, strip_diacritics: bool = False) -> str:
    """Run OCR on an image file path."""
    try:
        with open(file_path, "rb") as f:
            return perform_ocr_image_bytes(f.read(), strip_diacritics=strip_diacritics)
    except Exception as e:
        log.error(f"OCR image path error: {e}")
        return ""


def _ocr_max_workers(n_pages: int) -> int:
    return max(1, min(settings.OCR_MAX_CONCURRENT_PAGES, n_pages))


def perform_ocr_pdf_bytes(data: bytes, strip_diacritics: bool = False) -> str:
    """
    Convert every PDF page to an image and OCR it: bounded page-level
    parallelism (at most settings.OCR_MAX_CONCURRENT_PAGES pages — and
    therefore Tesseract subprocesses — running at once, regardless of how
    many pages the PDF has), tiered strategy escalation per page (see
    _ocr_image_tiered), and a page order that always matches the source
    PDF regardless of which page finishes OCR first. One page's exception
    is logged and that page is simply skipped — it never aborts the rest
    of the document.
    """
    try:
        pages = convert_from_bytes(data, dpi=200)
    except Exception as e:
        log.error(f"OCR PDF error (page rendering): {e}")
        return ""

    if not pages:
        return ""

    page_texts: List[Optional[str]] = [None] * len(pages)

    def _ocr_one_page(index: int, pil_img) -> None:
        try:
            img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
            text = _ocr_image_tiered(img, PDF_OCR_STRATEGIES, PDF_OCR_PSM_MODES)
            if text:
                page_texts[index] = f"[Page {index + 1}]\n{text}"
        except Exception as e:
            # Isolated per page: one bad page must not corrupt the rest of
            # the document's OCR output.
            log.warning(f"OCR PDF page {index + 1} failed, skipping just this page: {e}")

    max_workers = _ocr_max_workers(len(pages))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_ocr_one_page, i, p) for i, p in enumerate(pages)]
        for f in futures:
            f.result()  # re-raises only truly unexpected bugs — page-level errors are caught above

    # page_texts is indexed by page number, so joining in list order
    # preserves the original page order regardless of completion order.
    result = "\n\n".join(t for t in page_texts if t)
    if strip_diacritics:
        result = strip_arabic_diacritics(result)
    log.info(f"OCR PDF → {len(result)} chars from {len(pages)} page(s) (max_workers={max_workers})")
    return result


def perform_ocr_pdf_pages_bytes(
    data: bytes, page_indices: List[int], strip_diacritics: bool = False
) -> Dict[int, str]:
    """
    OCR only the given 0-based page indices of a PDF, instead of the whole
    document — used by loaders/pdf_loader.py for a MIXED text+scanned PDF,
    where only a handful of pages actually need OCR. Each requested page is
    rendered individually (first_page=last_page=that page), not the whole
    PDF, so this is cheap even for a large mostly-text document with only
    one or two scanned pages. Same bounded concurrency, tiered escalation,
    and per-page exception isolation as `perform_ocr_pdf_bytes`.

    Returns {page_index: text} — a page that fails or produces nothing is
    simply absent from the result, never a partial/corrupt entry.
    """
    if not page_indices:
        return {}

    results: Dict[int, str] = {}
    lock = threading.Lock()

    def _ocr_one(index: int) -> None:
        try:
            rendered = convert_from_bytes(data, dpi=200, first_page=index + 1, last_page=index + 1)
            if not rendered:
                return
            img = cv2.cvtColor(np.array(rendered[0].convert("RGB")), cv2.COLOR_RGB2BGR)
            text = _ocr_image_tiered(img, PDF_OCR_STRATEGIES, PDF_OCR_PSM_MODES)
            if strip_diacritics:
                text = strip_arabic_diacritics(text)
            if text:
                with lock:
                    results[index] = text
        except Exception as e:
            log.warning(f"OCR PDF page {index + 1} (mixed-document pass) failed, skipping: {e}")

    max_workers = _ocr_max_workers(len(page_indices))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_ocr_one, i) for i in page_indices]
        for f in futures:
            f.result()

    log.info(
        f"OCR PDF (mixed-document) → recovered {len(results)}/{len(page_indices)} "
        f"weak page(s) (max_workers={max_workers})"
    )
    return results


def perform_ocr_pdf_path(file_path: str, strip_diacritics: bool = False) -> str:
    try:
        with open(file_path, "rb") as f:
            return perform_ocr_pdf_bytes(f.read(), strip_diacritics=strip_diacritics)
    except Exception as e:
        log.error(f"OCR PDF path error: {e}")
        return ""


def extract_text(filename: str, data: bytes, strip_diacritics: bool = False) -> str:
    """Dispatch to the right extractor based on file extension."""
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return perform_ocr_pdf_bytes(data, strip_diacritics=strip_diacritics)
    if ext in {"png", "jpg", "jpeg", "tiff", "bmp", "webp"}:
        return perform_ocr_image_bytes(data, strip_diacritics=strip_diacritics)
    try:
        return data.decode("utf-8", errors="replace").strip()
    except Exception:
        return ""
