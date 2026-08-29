"""
evaluate_printed_ocr_arabic.py

Reproducible investigation of the printed-text Arabic OCR pipeline
(services/ocr_service.py, Tesseract + OpenCV) — separate from the
already-evaluated handwritten OCR path (evaluate_handwritten_ocr.py).

What this script does:
  1. Renders a small synthetic Arabic test corpus with KNOWN ground truth,
     covering the specific failure modes commonly reported for Arabic OCR:
     diacritics (tashkeel), the لا ligature, visually-confusable letters,
     Arabic-Indic vs. Western digits, a longer prose paragraph, and mixed
     Arabic/English text — across several fonts (the project's own bundled
     Amiri, plus Windows Arial/Tahoma/Times New Roman/Arabic Typesetting),
     and several image-quality degradations (low-res, skew, noise, low
     contrast) simulating real scans/photos.

     No real scanned/printed Arabic document corpus was found in this repo
     or readily available locally, so this uses the SAME technique the
     project's own report_service.py already relies on to render Arabic
     (arabic_reshaper + python-bidi) to produce clean, correctly-shaped
     ground-truth images — a defensible synthetic-but-faithful substitute,
     same approach evaluate_handwritten_ocr.py used for its one synthetic
     mixed-language sample. Treat absolute CER/WER numbers here as
     indicative of the engine/config's ceiling on clean-to-degraded
     renders, not a certified real-world benchmark.

  2. Runs the CURRENT production pipeline (services.ocr_service) against
     every image, computes CER/WER against ground truth, and prints
     character-level diffs for the worst results.

  3. Reproduces, with real evidence (not assumption), a critical
     deployment-config bug: backend/Dockerfile's `apt-get install
     tesseract-ocr` never installs `tesseract-ocr-ara`, so `ara.traineddata`
     is ABSENT in the actual Docker image this project ships (verified
     against Debian's package metadata: tesseract-ocr's only language
     dependencies are tesseract-ocr-eng + tesseract-ocr-osd). This is
     simulated locally by pointing TESSDATA_PREFIX at a directory that
     only has eng+osd data (mirroring the Docker image exactly) and running
     the exact same `-l ara+eng` config the app uses.

Run:
    python scripts/evaluate_printed_ocr_arabic.py
Requires a local Tesseract install with `ara` traineddata (already present
on this machine at C:\\Program Files\\Tesseract-OCR).
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.isfile(TESSERACT_EXE):
    os.environ.setdefault("TESSERACT_CMD", TESSERACT_EXE)

import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
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


# ── CER / WER (Levenshtein) ──────────────────────────────────────────────

def _edit_distance(a: List[str], b: List[str]) -> int:
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


def cer(reference: str, hypothesis: str) -> float:
    ref = list(reference.strip())
    if not ref:
        return 0.0 if not hypothesis.strip() else 1.0
    return _edit_distance(ref, list(hypothesis.strip())) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    ref = reference.strip().split()
    if not ref:
        return 0.0 if not hypothesis.strip() else 1.0
    return _edit_distance(ref, hypothesis.strip().split()) / len(ref)


def char_diff(reference: str, hypothesis: str) -> str:
    """Simple aligned diff for human inspection (not the CER computation
    itself — just a readable side-by-side of the two strings)."""
    r, h = reference.strip(), hypothesis.strip()
    return f"    GT : {r}\n    OUT: {h}"


# ── Ground truth corpus ──────────────────────────────────────────────────

@dataclass
class Sample:
    id: str
    category: str
    text: str


CORPUS: List[Sample] = [
    Sample("s1_plain", "plain, no diacritics",
           "مرحبا بكم في هذا المستند التقني الخاص بالمشروع"),
    Sample("s2_diacritics", "full diacritics (tashkeel)",
           "الْعِلْمُ نُورٌ وَالْجَهْلُ ظَلَامٌ فِي كُلِّ زَمَانٍ وَمَكَانٍ"),
    Sample("s3_ligature_laa", "لا ligature heavy",
           "لا إله إلا الله ولا حول ولا قوة إلا بالله العلي العظيم"),
    Sample("s4_confusable_letters", "visually-similar letters (ب ت ث / ح ج خ / س ش / ص ض / ط ظ / ع غ / ف ق)",
           "الجو حار وهناك خيول وجبال، سوق شعبي مزدحم بالناس، صف طويل من الطلاب الغيورين على الفقراء"),
    Sample("s5_digits", "Arabic-Indic vs Western digits",
           "رقم الهاتف ٠١٢٣٤٥٦٧٨٩ ورقم آخر بالأرقام الغربية 0123456789"),
    Sample("s6_date_number", "date + reference number in a sentence",
           "تم إصدار هذا التقرير بتاريخ ٢٩ أغسطس ٢٠٢٦ برقم مرجعي 4587"),
    Sample("s7_paragraph", "longer prose paragraph (~50 words)",
           "يهدف هذا المشروع إلى تطوير نظام ذكي قادر على فهم المستندات العربية "
           "والإنجليزية والإجابة عن أسئلة المستخدمين بدقة عالية. يعتمد النظام على "
           "تقنيات معالجة اللغة الطبيعية والتعلم الآلي لاستخراج المعلومات من الملفات "
           "المرفوعة، ويقوم بتحليل النصوص وتلخيصها بشكل تلقائي لمساعدة الطلاب "
           "والباحثين على الوصول إلى المعلومة بسرعة وسهولة."),
    Sample("s8_mixed", "mixed Arabic + English (technical)",
           "نظام إدارة قواعد البيانات Database Management System يُستخدم على نطاق "
           "واسع في الشركات الكبرى مثل Oracle و MySQL."),
]

FONT_CANDIDATES = {
    "Amiri (bundled, used by report_service.py)": BACKEND_DIR / "assets" / "fonts" / "Amiri-Regular.ttf",
    "Arial": Path(r"C:\Windows\Fonts\arial.ttf"),
    "Tahoma": Path(r"C:\Windows\Fonts\tahoma.ttf"),
    "Times New Roman": Path(r"C:\Windows\Fonts\times.ttf"),
    "Arabic Typesetting": Path(r"C:\Windows\Fonts\arabtype.ttf"),
}
FONTS = {name: path for name, path in FONT_CANDIDATES.items() if path.is_file()}


# ── Rendering ─────────────────────────────────────────────────────────────

def render_arabic(text: str, font_path: Path, font_size: int = 40,
                   margin: int = 30) -> Image.Image:
    """Render text (Arabic reshaped+bidi-reordered exactly like
    report_service._shape does; English/digit runs pass through unchanged
    inside arabic_reshaper/get_display's own mixed-script handling) onto a
    white canvas using a real Arabic-capable font. PIL/Pillow here has no
    libraqm, so shaping must be done ourselves before drawing — same
    reason report_service.py does it for PDF rendering."""
    shaped = get_display(arabic_reshaper.reshape(text))
    font = ImageFont.truetype(str(font_path), font_size)

    tmp = Image.new("RGB", (10, 10), "white")
    bbox = ImageDraw.Draw(tmp).textbbox((0, 0), shaped, font=font)
    w = (bbox[2] - bbox[0]) + margin * 2
    h = (bbox[3] - bbox[1]) + margin * 2

    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin - bbox[0], margin - bbox[1]), shaped, font=font, fill="black")
    return img


def degrade_low_res(img: Image.Image) -> Image.Image:
    """Simulate a low-resolution scan/photo: downscale hard, then upscale
    back with a crude resample so detail is genuinely lost (not just
    resized)."""
    w, h = img.size
    small = img.resize((max(1, w // 4), max(1, h // 4)), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def degrade_skew(img: Image.Image, angle: float = 6.0) -> Image.Image:
    return img.rotate(angle, expand=True, fillcolor="white", resample=Image.BICUBIC)


def degrade_noise(img: Image.Image) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.default_rng(7).normal(0, 22, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    noisy = Image.fromarray(arr)
    return noisy.filter(__import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(radius=0.8))


def degrade_low_contrast(img: Image.Image) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(0.35)


# ── OCR runner (imports the ACTUAL production pipeline, unmodified) ────────

def run_current_pipeline(img: Image.Image) -> str:
    from services.ocr_service import perform_ocr_image_bytes
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return perform_ocr_image_bytes(buf.getvalue())


def run_raw_tesseract(img: Image.Image, lang: str, psm: int = 6, oem: int = 1) -> str:
    """Bypass the app's preprocessing sweep entirely — raw image straight
    into pytesseract with the app's exact `-l`/`--oem`/`--psm` config, to
    isolate "is preprocessing helping/hurting" from "is the engine/config
    itself the ceiling"."""
    config = f"--oem {oem} --psm {psm} -l {lang}"
    return pytesseract.image_to_string(img, config=config).strip()


# ── Docker-deployment bug reproduction ──────────────────────────────────────

def build_fake_docker_tessdata() -> Path:
    """Mirror exactly what `apt-get install tesseract-ocr` (backend/
    Dockerfile line 38, no `tesseract-ocr-ara`) actually provides on
    Debian bookworm: eng + osd traineddata, NO ara. Verified against
    Debian's package metadata (tesseract-ocr Depends: tesseract-ocr-eng,
    tesseract-ocr-osd — ara is a separate, not-installed package)."""
    real_tessdata = Path(pytesseract.pytesseract.tesseract_cmd).parent / "tessdata"
    fake_dir = SCRATCH / "fake_docker_tessdata"
    fake_dir.mkdir(parents=True, exist_ok=True)
    for name in ("eng.traineddata", "osd.traineddata"):
        src = real_tessdata / name
        if src.is_file():
            shutil.copyfile(src, fake_dir / name)
    return fake_dir


def reproduce_docker_bug(sample_img: Image.Image) -> dict:
    fake_tessdata = build_fake_docker_tessdata()
    present = sorted(p.name for p in fake_tessdata.glob("*.traineddata"))

    env_backup = os.environ.get("TESSDATA_PREFIX")
    os.environ["TESSDATA_PREFIX"] = str(fake_tessdata)
    try:
        try:
            text = pytesseract.image_to_string(
                sample_img, config="--oem 1 --psm 6 -l ara+eng"
            ).strip()
            error = None
        except Exception as e:
            text = None
            error = f"{type(e).__name__}: {e}"
    finally:
        if env_backup is None:
            os.environ.pop("TESSDATA_PREFIX", None)
        else:
            os.environ["TESSDATA_PREFIX"] = env_backup

    return {"fake_tessdata_contents": present, "output_text": text, "exception": error}


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 100)
    print("PRINTED ARABIC OCR EVALUATION (services/ocr_service.py, Tesseract)")
    print("=" * 100)
    print(f"tesseract_cmd = {pytesseract.pytesseract.tesseract_cmd}")
    try:
        print(f"tesseract version = {pytesseract.get_tesseract_version()}")
        print(f"available languages = {pytesseract.get_languages()}")
    except Exception as e:
        print(f"Could not query tesseract: {e}")
    print(f"Fonts available for rendering: {list(FONTS.keys())}")

    results = []

    # ── Part A: core corpus, default font (Amiri — matches what a printed
    # Arabic PDF/document most plausibly looks like), current production
    # pipeline vs raw single-pass tesseract with the app's exact config. ──
    print("\n" + "-" * 100)
    print("PART A — Core failure-mode corpus (font=Amiri), current pipeline")
    print("-" * 100)
    amiri = FONTS.get("Amiri (bundled, used by report_service.py)")
    for s in CORPUS:
        img = render_arabic(s.text, amiri)
        img.save(IMG_DIR / f"{s.id}_amiri.png")

        t0 = time.perf_counter()
        pipeline_text = run_current_pipeline(img)
        pipeline_time = time.perf_counter() - t0
        raw_text = run_raw_tesseract(img, "ara+eng", psm=6, oem=1)

        c_pipe, w_pipe = cer(s.text, pipeline_text), wer(s.text, pipeline_text)
        c_raw, w_raw = cer(s.text, raw_text), wer(s.text, raw_text)

        print(f"\n[{s.id}] {s.category}")
        print(f"  pipeline: CER={c_pipe:.3f} WER={w_pipe:.3f} time={pipeline_time:.2f}s")
        print(char_diff(s.text, pipeline_text))
        print(f"  raw tesseract (single pass, no preprocessing sweep): CER={c_raw:.3f} WER={w_raw:.3f}")
        print(char_diff(s.text, raw_text))

        results.append({
            "part": "A", "id": s.id, "category": s.category, "font": "Amiri",
            "ground_truth": s.text,
            "pipeline_output": pipeline_text, "pipeline_cer": round(c_pipe, 3), "pipeline_wer": round(w_pipe, 3),
            "pipeline_time_s": round(pipeline_time, 2),
            "raw_output": raw_text, "raw_cer": round(c_raw, 3), "raw_wer": round(w_raw, 3),
        })

    # ── Part B: font sensitivity — s7 paragraph across every available font ──
    print("\n" + "-" * 100)
    print("PART B — Font sensitivity (s7 paragraph, current pipeline)")
    print("-" * 100)
    s7 = next(s for s in CORPUS if s.id == "s7_paragraph")
    for font_name, font_path in FONTS.items():
        img = render_arabic(s7.text, font_path)
        safe_name = font_name.split(" ")[0].lower()
        img.save(IMG_DIR / f"s7_font_{safe_name}.png")
        text = run_current_pipeline(img)
        c, w = cer(s7.text, text), wer(s7.text, text)
        print(f"\n[{font_name}] CER={c:.3f} WER={w:.3f}")
        print(char_diff(s7.text, text))
        results.append({"part": "B", "id": "s7_paragraph", "font": font_name,
                         "ground_truth": s7.text, "pipeline_output": text,
                         "pipeline_cer": round(c, 3), "pipeline_wer": round(w, 3)})

    # ── Part C: image-quality degradations — s7 paragraph, Amiri font ──────
    print("\n" + "-" * 100)
    print("PART C — Image quality degradations (s7 paragraph, font=Amiri, current pipeline)")
    print("-" * 100)
    clean_img = render_arabic(s7.text, amiri)
    degradations = {
        "clean (baseline)": lambda im: im,
        "low_resolution": degrade_low_res,
        "skewed_6deg": degrade_skew,
        "noisy": degrade_noise,
        "low_contrast": degrade_low_contrast,
    }
    for name, fn in degradations.items():
        img = fn(clean_img)
        img.save(IMG_DIR / f"s7_quality_{name.split()[0]}.png")
        t0 = time.perf_counter()
        text = run_current_pipeline(img)
        elapsed = time.perf_counter() - t0
        c, w = cer(s7.text, text), wer(s7.text, text)
        print(f"\n[{name}] CER={c:.3f} WER={w:.3f} time={elapsed:.2f}s")
        print(char_diff(s7.text, text))
        results.append({"part": "C", "id": "s7_paragraph", "degradation": name,
                         "ground_truth": s7.text, "pipeline_output": text,
                         "pipeline_cer": round(c, 3), "pipeline_wer": round(w, 3),
                         "pipeline_time_s": round(elapsed, 2)})

    # ── Part D: reproduce the Docker-deployment missing-ara-traineddata bug ─
    print("\n" + "-" * 100)
    print("PART D — Reproducing backend/Dockerfile's missing tesseract-ocr-ara")
    print("-" * 100)
    s1_img = render_arabic(CORPUS[0].text, amiri)
    dbug = reproduce_docker_bug(s1_img)
    print(f"  Simulated container tessdata/ contents: {dbug['fake_tessdata_contents']}")
    print(f"  Running app's exact config '--oem 1 --psm 6 -l ara+eng' against this tessdata...")
    print(f"  Exception raised: {dbug['exception']}")
    print(f"  Output text: {dbug['output_text']!r}")
    results.append({"part": "D", "description": "Docker-image tessdata simulation (no ara.traineddata)", **dbug})

    out_path = SCRATCH / "printed_ocr_eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nFull results written to {out_path}")
    print(f"Rendered images written to {IMG_DIR}")


if __name__ == "__main__":
    main()
