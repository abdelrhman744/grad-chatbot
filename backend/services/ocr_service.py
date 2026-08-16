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
    try:
        processed = _preprocess_for_ocr(img, best_strategy)
        first_text = _run_tesseract(processed, best_psm)
    except Exception as e:
        log.debug(f"tiered OCR first pass failed (strategy={best_strategy}, psm={best_psm}): {e}")

    if _ocr_result_confident(first_text):
        return first_text

    # Escalate: run every other combination too and merge everything,
    # exactly like the original always-run-all behavior — this only fires
    # for pages/images that actually need it.
    results = [first_text] if first_text else []
    for strategy in strategies:
        for psm in psm_modes:
            if strategy == best_strategy and psm == best_psm:
                continue
            try:
                processed = _preprocess_for_ocr(img, strategy)
                text = _run_tesseract(processed, psm)
                if text:
                    results.append(text)
            except Exception:
                continue
    return _merge_ocr_results(results)


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

def perform_ocr_image_bytes(data: bytes) -> str:
    """Run OCR on raw image bytes (tiered — see _ocr_image_tiered)."""
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
        except Exception:
            pass

    log.info(f"OCR image → {len(merged)} chars")
    return merged


def perform_ocr_image_path(file_path: str) -> str:
    """Run OCR on an image file path."""
    try:
        with open(file_path, "rb") as f:
            return perform_ocr_image_bytes(f.read())
    except Exception as e:
        log.error(f"OCR image path error: {e}")
        return ""


def _ocr_max_workers(n_pages: int) -> int:
    return max(1, min(settings.OCR_MAX_CONCURRENT_PAGES, n_pages))


def perform_ocr_pdf_bytes(data: bytes) -> str:
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
    log.info(f"OCR PDF → {len(result)} chars from {len(pages)} page(s) (max_workers={max_workers})")
    return result


def perform_ocr_pdf_pages_bytes(data: bytes, page_indices: List[int]) -> Dict[int, str]:
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


def perform_ocr_pdf_path(file_path: str) -> str:
    try:
        with open(file_path, "rb") as f:
            return perform_ocr_pdf_bytes(f.read())
    except Exception as e:
        log.error(f"OCR PDF path error: {e}")
        return ""


def extract_text(filename: str, data: bytes) -> str:
    """Dispatch to the right extractor based on file extension."""
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return perform_ocr_pdf_bytes(data)
    if ext in {"png", "jpg", "jpeg", "tiff", "bmp", "webp"}:
        return perform_ocr_image_bytes(data)
    try:
        return data.decode("utf-8", errors="replace").strip()
    except Exception:
        return ""
