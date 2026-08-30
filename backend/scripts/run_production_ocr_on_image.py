"""
run_production_ocr_on_image.py

Runs the ACTUAL production OCR entry point — services.ocr_service.
perform_ocr_image_bytes, unmodified — against one real image file and
prints the literal raw result. No alternate/bypass OCR implementation is
used anywhere in this script.

The only non-production code here is instrumentation: thin wrappers around
a few of ocr_service's own internal functions (_run_tesseract,
_run_tesseract_digit_whitelist, _postprocess_ocr_text) that log what they
were called with / returned, then call straight through to the real
function. This exists purely so this run's terminal output can show which
(strategy, PSM) combination actually fired and what post-processing
changed — perform_ocr_image_bytes's own control flow and logic are never
altered or re-implemented.

Usage:
    python scripts/run_production_ocr_on_image.py "<path to image>"
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.isfile(TESSERACT_EXE):
    os.environ.setdefault("TESSERACT_CMD", TESSERACT_EXE)

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

from services import ocr_service  # noqa: E402

_calls = []  # records of every _run_tesseract call: (strategy_or_None, psm, result_len)


def _instrument():
    """Wrap (not replace) ocr_service's own internal functions purely to
    record what happened — every wrapper calls the real, original
    function and returns its real result unchanged."""
    real_run_tesseract = ocr_service._run_tesseract

    def spy_run_tesseract(img, psm=6):
        text = real_run_tesseract(img, psm)
        _calls.append({"fn": "_run_tesseract", "psm": psm, "config": f"--oem 1 --psm {psm} -l ara+eng",
                        "result_len": len(text), "result_preview": text[:80]})
        return text

    real_digit_whitelist = ocr_service._run_tesseract_digit_whitelist

    def spy_digit_whitelist(img):
        text = real_digit_whitelist(img)
        _calls.append({"fn": "_run_tesseract_digit_whitelist", "result_len": len(text),
                        "result_preview": text[:80]})
        return text

    real_postprocess = ocr_service._postprocess_ocr_text

    def spy_postprocess(text, img=None):
        before = text
        after = real_postprocess(text, img)
        _calls.append({"fn": "_postprocess_ocr_text", "changed": before != after,
                        "before_len": len(before) if before else 0,
                        "after_len": len(after) if after else 0})
        return after

    ocr_service._run_tesseract = spy_run_tesseract
    ocr_service._run_tesseract_digit_whitelist = spy_digit_whitelist
    ocr_service._postprocess_ocr_text = spy_postprocess


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_production_ocr_on_image.py <image path>")
        sys.exit(1)
    image_path = sys.argv[1]

    print("=" * 100)
    print("STEP 1 — engine / config identification (as actually configured in this process)")
    print("=" * 100)
    import pytesseract
    print(f"tesseract_cmd     = {pytesseract.pytesseract.tesseract_cmd}")
    print(f"tesseract version = {pytesseract.get_tesseract_version()}")
    print(f"available langs   = {pytesseract.get_languages()}")
    print(f"OCR_STRATEGIES    = {ocr_service.OCR_STRATEGIES}")
    print(f"OCR_PSM_MODES     = {ocr_service.OCR_PSM_MODES}")
    print(f"base config       = --oem 1 --psm <mode> -l ara+eng  (see _run_tesseract)")

    print("\n" + "=" * 100)
    print(f"STEP 2 — reading image: {image_path}")
    print("=" * 100)
    with open(image_path, "rb") as f:
        data = f.read()
    print(f"File size: {len(data)} bytes")

    _instrument()

    print("\n" + "=" * 100)
    print("STEP 3 — calling the ACTUAL production function: "
          "services.ocr_service.perform_ocr_image_bytes(data)")
    print("=" * 100)
    result = ocr_service.perform_ocr_image_bytes(data)

    print("\n" + "=" * 100)
    print("STEP 4 — internal call trace (which strategy/PSM combo(s) actually ran)")
    print("=" * 100)
    for i, c in enumerate(_calls):
        print(f"  [{i}] {c}")

    print("\n" + "=" * 100)
    print("STEP 5 — RAW OCR OUTPUT (verbatim, repr() to show every character exactly)")
    print("=" * 100)
    print(repr(result))
    print("\n--- same text, rendered directly (not repr) ---")
    print(result)
    print(f"\nLength: {len(result)} chars, {len(result.split())} words")


if __name__ == "__main__":
    main()
