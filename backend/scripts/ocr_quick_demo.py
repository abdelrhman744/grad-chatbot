"""
ocr_quick_demo.py — fast, minimal, real-output demo of the printed Arabic
OCR pipeline (services/ocr_service.py, Tesseract). Trimmed down from
evaluate_printed_ocr_arabic.py's full sweep (which is slow on Windows due
to per-subprocess Tesseract startup cost) to just 3 samples + the
Docker-deployment bug reproduction, so it finishes in well under a minute
and prints real, unsummarized output directly to stdout.
"""
from __future__ import annotations

import io
import os
import shutil
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.isfile(TESSERACT_EXE):
    os.environ.setdefault("TESSERACT_CMD", TESSERACT_EXE)

import pytesseract
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

pytesseract.pytesseract.tesseract_cmd = os.environ.get("TESSERACT_CMD", "tesseract")

SCRATCH = Path(os.environ.get(
    "OCR_EVAL_SCRATCH",
    str(Path.home() / "AppData/Local/Temp/claude/c--Graduation-Agentic-AI-grad-chatbot"
        "/61d69e02-4d9c-4426-89ea-fb707688bc42/scratchpad/ocr_investigation"),
))
IMG_DIR = SCRATCH / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def _edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def cer(reference, hypothesis, strip_diacritics=False):
    """strip_diacritics=True removes Arabic tashkeel from BOTH strings
    before comparing (services.ocr_service.strip_arabic_diacritics) — most
    RAG/search use cases don't need tashkeel, so a diacritic Tesseract
    dropped shouldn't count as a "real" error for those callers. Off by
    default so the raw, diacritics-sensitive number is still available."""
    if strip_diacritics:
        from services.ocr_service import strip_arabic_diacritics
        reference, hypothesis = strip_arabic_diacritics(reference), strip_arabic_diacritics(hypothesis)
    ref = list(reference.strip())
    if not ref:
        return 0.0 if not hypothesis.strip() else 1.0
    return _edit_distance(ref, list(hypothesis.strip())) / len(ref)


def wer(reference, hypothesis, strip_diacritics=False):
    if strip_diacritics:
        from services.ocr_service import strip_arabic_diacritics
        reference, hypothesis = strip_arabic_diacritics(reference), strip_arabic_diacritics(hypothesis)
    ref = reference.strip().split()
    if not ref:
        return 0.0 if not hypothesis.strip() else 1.0
    return _edit_distance(ref, hypothesis.strip().split()) / len(ref)


AMIRI = BACKEND_DIR / "assets" / "fonts" / "Amiri-Regular.ttf"

SAMPLES = [
    ("s1_plain", "مرحبا بكم في هذا المستند التقني الخاص بالمشروع"),
    ("s2_diacritics", "الْعِلْمُ نُورٌ وَالْجَهْلُ ظَلَامٌ فِي كُلِّ زَمَانٍ وَمَكَانٍ"),
    ("s3_ligature_laa", "لا إله إلا الله ولا حول ولا قوة إلا بالله العلي العظيم"),
    ("s5_digits", "رقم الهاتف ٠١٢٣٤٥٦٧٨٩ ورقم آخر بالأرقام الغربية 0123456789"),
]


def render(text, font_path=AMIRI, size=40, margin=30):
    shaped = get_display(arabic_reshaper.reshape(text))
    font = ImageFont.truetype(str(font_path), size)
    tmp = Image.new("RGB", (10, 10), "white")
    bbox = ImageDraw.Draw(tmp).textbbox((0, 0), shaped, font=font)
    w, h = (bbox[2] - bbox[0]) + margin * 2, (bbox[3] - bbox[1]) + margin * 2
    img = Image.new("RGB", (w, h), "white")
    ImageDraw.Draw(img).text((margin - bbox[0], margin - bbox[1]), shaped, font=font, fill="black")
    return img


def main():
    print("=" * 90)
    print("STEP 1 — engine identification")
    print("=" * 90)
    print(f"tesseract_cmd = {pytesseract.pytesseract.tesseract_cmd}")
    print(f"tesseract version = {pytesseract.get_tesseract_version()}")
    langs = pytesseract.get_languages()
    print(f"available languages on THIS machine = {langs}")

    from services.ocr_service import perform_ocr_image_bytes

    print("\n" + "=" * 90)
    print("STEP 2 — run the ACTUAL production pipeline (services.ocr_service."
          "perform_ocr_image_bytes) on real Arabic images")
    print("=" * 90)

    for sid, text in SAMPLES:
        img = render(text)
        path = IMG_DIR / f"{sid}.png"
        img.save(path)
        print(f"\n--- sample: {sid}  (image saved to {path}, size={img.size}) ---")
        print(f"GROUND TRUTH : {text}")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        t0 = time.perf_counter()
        result = perform_ocr_image_bytes(buf.getvalue())
        elapsed = time.perf_counter() - t0

        print(f"RAW OCR OUTPUT (verbatim, {elapsed:.2f}s): {result!r}")
        c, w = cer(text, result), wer(text, result)
        print(f"CER={c:.3f}  WER={w:.3f}")
        if sid == "s2_diacritics":
            c_nd, w_nd = cer(text, result, strip_diacritics=True), wer(text, result, strip_diacritics=True)
            print(f"CER (diacritics-insensitive comparison)={c_nd:.3f}  WER={w_nd:.3f}")

    print("\n" + "=" * 90)
    print("STEP 3 — reproduce backend/Dockerfile's missing `tesseract-ocr-ara` package")
    print("=" * 90)
    real_tessdata = Path(pytesseract.pytesseract.tesseract_cmd).parent / "tessdata"
    fake_dir = SCRATCH / "fake_docker_tessdata"
    fake_dir.mkdir(parents=True, exist_ok=True)
    for name in ("eng.traineddata", "osd.traineddata"):
        src = real_tessdata / name
        if src.is_file():
            shutil.copyfile(src, fake_dir / name)
    print(f"Simulated container tessdata dir: {fake_dir}")
    print(f"Contents (mirrors `apt-get install tesseract-ocr` with NO "
          f"tesseract-ocr-ara, exactly as backend/Dockerfile does today): "
          f"{sorted(p.name for p in fake_dir.glob('*.traineddata'))}")

    img = render(SAMPLES[0][1])
    env_backup = os.environ.get("TESSDATA_PREFIX")
    os.environ["TESSDATA_PREFIX"] = str(fake_dir)
    try:
        try:
            out = pytesseract.image_to_string(img, config="--oem 1 --psm 6 -l ara+eng").strip()
            print(f"Running app's exact config '--oem 1 --psm 6 -l ara+eng' -> output: {out!r}")
        except Exception as e:
            print(f"Running app's exact config '--oem 1 --psm 6 -l ara+eng' -> "
                  f"RAISED {type(e).__name__}: {e}")
    finally:
        if env_backup is None:
            os.environ.pop("TESSDATA_PREFIX", None)
        else:
            os.environ["TESSDATA_PREFIX"] = env_backup

    print("\nDone.")


if __name__ == "__main__":
    main()
