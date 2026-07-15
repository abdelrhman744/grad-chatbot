import logging
import re
import numpy as np
import cv2
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes, convert_from_path
from typing import List

from config import settings

log = logging.getLogger("ocr_service")

pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD

OCR_STRATEGIES = ["adaptive", "otsu", "denoise", "sharpen", "contrast"]
OCR_PSM_MODES  = [6, 3, 11]


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


# ── Public API ─────────────────────────────────────────────────────────────────

def perform_ocr_image_bytes(data: bytes) -> str:
    """Run OCR on raw image bytes."""
    results = []
    try:
        nparr = np.frombuffer(data, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            from PIL import Image as PILImage
            import io
            pil = PILImage.open(io.BytesIO(data)).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        for strategy in OCR_STRATEGIES:
            try:
                processed = _preprocess_for_ocr(img, strategy)
                for psm in OCR_PSM_MODES:
                    text = _run_tesseract(processed, psm)
                    if text:
                        results.append(text)
            except Exception:
                continue
    except Exception as e:
        log.error(f"OCR image error: {e}")

    if not results:
        try:
            import io
            results.append(pytesseract.image_to_string(
                Image.open(io.BytesIO(data)), lang="ara+eng", config="--oem 1 --psm 6"
            ).strip())
        except Exception:
            pass

    merged = _merge_ocr_results(results)
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


def perform_ocr_pdf_bytes(data: bytes) -> str:
    """Convert PDF pages to images and OCR each one."""
    try:
        pages = convert_from_bytes(data, dpi=200)
        page_texts = []
        for i, pil_img in enumerate(pages):
            img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
            results = []
            for strategy in ["adaptive", "otsu", "denoise"]:
                try:
                    processed = _preprocess_for_ocr(img, strategy)
                    for psm in [6, 3]:
                        text = _run_tesseract(processed, psm)
                        if text:
                            results.append(text)
                except Exception:
                    continue
            page_text = _merge_ocr_results(results)
            if page_text:
                page_texts.append(f"[Page {i+1}]\n{page_text}")
        result = "\n\n".join(page_texts)
        log.info(f"OCR PDF → {len(result)} chars")
        return result
    except Exception as e:
        log.error(f"OCR PDF error: {e}")
        return ""


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
