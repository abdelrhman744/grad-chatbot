"""
evaluate_handwritten_ocr.py

Tasks 2 & 6 — reproducible evaluation harness for the handwritten OCR
service (services/handwritten_ocr_service.py).

Data sources (see the final report for full provenance/caveats):
  - English: REAL handwriting from the Teklia/IAM-line dataset (the
    standard IAM handwriting line-recognition benchmark), already cached
    locally on this machine.
  - Arabic: REAL handwriting images from the KHATT corpus
    (johnlockejrr/KHATT_v1.0_dataset test split), paired with ground
    truth from that same dataset's `config_files.tar.bz2` (test_ids.txt /
    test_text.txt). NOTE: that ground-truth text file stores each line's
    characters in REVERSED order relative to Unicode logical order
    (verified by inspection — reversing turns gibberish into grammatical
    Arabic); this script reverses it back. This reversal was verified by
    manual inspection of several samples, not by an authoritative KHATT
    format spec, so treat Arabic CER/WER here as good-but-not-certified.
  - Mixed Arabic/English single line: NO real mixed-handwriting dataset
    was found anywhere accessible; this one sample is synthetically
    rendered (regular, non-cursive fonts) and clearly excluded from the
    "real handwriting" comparison table — reported separately, caveated.

Run:
    python scripts/evaluate_handwritten_ocr.py
Requires backend/requirements-dev.txt (psutil) installed; downloads two
small real datasets on first run (a few hundred KB each, cached after).
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:
    import psutil
    _process = psutil.Process(os.getpid())
except ImportError:
    psutil = None
    _process = None


# ── CER / WER (hand-written Levenshtein — no jiwer/python-Levenshtein dep) ──

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


def rss_mb() -> Optional[float]:
    if _process is None:
        return None
    return _process.memory_info().rss / (1024 * 1024)


# ── Sample construction ─────────────────────────────────────────────────

@dataclass
class Sample:
    id: str
    lang: str  # "ar" | "en" | "mixed"
    kind: str  # "real" | "synthetic"
    image: Image.Image
    ground_truth: str
    note: str = ""


def _find_english_handwriting_font() -> Optional[str]:
    candidates = [
        r"C:\Windows\Fonts\segoescr.ttf",   # Segoe Script
        r"C:\Windows\Fonts\SCRIPTBL.TTF",
        r"C:\Windows\Fonts\FREESCPT.TTF",
        r"C:\Windows\Fonts\MISTRAL.TTF",
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def load_real_english_samples(n: int = 6) -> List[Sample]:
    from datasets import load_dataset

    ds = load_dataset("Teklia/IAM-line", split="test")
    lengths = sorted(range(len(ds)), key=lambda i: len(ds[i]["text"]))
    # short, medium, long spread instead of the first N (avoids bias toward
    # whatever happens to sort first).
    picks = [lengths[i] for i in np.linspace(0, len(lengths) - 1, n).astype(int)]
    samples = []
    for idx in picks:
        ex = ds[int(idx)]
        samples.append(
            Sample(id=f"iam-line-{idx}", lang="en", kind="real",
                   image=ex["image"].convert("RGB"), ground_truth=ex["text"].strip())
        )
    return samples


def load_real_arabic_samples(manifest_path: str) -> List[Sample]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    samples = []
    for entry in manifest:
        img = Image.open(entry["local_path"]).convert("RGB")
        samples.append(
            Sample(id=entry["id"], lang="ar", kind="real",
                   image=img, ground_truth=entry["text"].strip())
        )
    return samples


def make_mixed_synthetic_sample(font_path: Optional[str]) -> Sample:
    text = "Course CS201 نظم التشغيل Operating Systems"
    font = ImageFont.truetype(font_path, 40) if font_path else ImageFont.load_default()
    img = Image.new("RGB", (1000, 90), "white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 15), text, font=font, fill="black")
    return Sample(id="mixed-synthetic-1", lang="mixed", kind="synthetic", image=img,
                  ground_truth=text,
                  note="Synthetic, regular (non-cursive) font — no real mixed AR/EN handwriting dataset was found.")


def make_noisy_variant(sample: Sample) -> Sample:
    img = sample.image.filter(ImageFilter.GaussianBlur(radius=1.2))
    arr = np.array(img).astype(np.float32)
    noise = np.random.default_rng(42).normal(0, 18, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    noisy = Image.fromarray(arr)
    return Sample(id=sample.id + "-noisy", lang=sample.lang, kind=sample.kind, image=noisy,
                  ground_truth=sample.ground_truth, note="Gaussian blur + noise applied to a real sample.")


def make_multiline_page(samples: List[Sample], lang: str) -> Sample:
    """Stack several real single-line images into one page-like image, so
    line segmentation can be evaluated against genuine handwriting content
    instead of only single pre-cropped lines."""
    imgs = [s.image for s in samples]
    width = max(im.width for im in imgs) + 40
    gap = 30
    height = sum(im.height for im in imgs) + gap * (len(imgs) + 1)
    page = Image.new("RGB", (width, height), "white")
    y = gap
    for im in imgs:
        page.paste(im, (20, y))
        y += im.height + gap
    combined_text = "\n".join(s.ground_truth for s in samples)
    return Sample(id=f"multiline-{lang}", lang=lang, kind="real", image=page,
                  ground_truth=combined_text, note=f"{len(samples)} real lines stacked into one page.")


# ── Benchmark runners ────────────────────────────────────────────────────

def run_sequential(service, image: Image.Image, lang: str):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    t0 = time.perf_counter()
    text, debug = service.recognize_with_debug(buf.getvalue(), lang)
    elapsed = time.perf_counter() - t0
    return text, elapsed, debug


def run_batched(service, image: Image.Image, lang: str, max_batch: int = 16):
    """Standalone batched-inference variant, NOT the production code path —
    used here only to measure whether batching is worth adopting. Mirrors
    handwritten_ocr_service._segment_lines + model.generate but feeds every
    line through the model in one (or a few, RAM-bounded) forward pass(es)
    instead of one call per line."""
    import torch

    from services.handwritten_ocr_service import _preprocess, _segment_lines

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    prepped = _preprocess(buf.getvalue())

    t0 = time.perf_counter()
    lines = _segment_lines(prepped)
    t1 = time.perf_counter()

    processor, model = service._get_model(lang)
    device = service._resolve_device()

    texts = []
    for i in range(0, len(lines), max_batch):
        batch = lines[i:i + max_batch]
        pixel_values = processor(images=batch, return_tensors="pt").pixel_values.to(device)
        eos = model.config.eos_token_id or model.generation_config.eos_token_id
        pad = model.config.pad_token_id or model.generation_config.pad_token_id
        with torch.inference_mode():
            generated_ids = model.generate(
                pixel_values, max_new_tokens=256, eos_token_id=eos, pad_token_id=pad, use_cache=True,
            )
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
        texts.extend(t.strip() for t in decoded)
    t2 = time.perf_counter()

    combined = "\n".join(t for t in texts if t)
    elapsed = t2 - t0
    return combined, elapsed, {"num_lines": len(lines), "segment_time_s": t1 - t0, "inference_time_s": t2 - t1}


def run_no_segmentation(service, image: Image.Image, lang: str):
    """Feed the WHOLE multi-line page directly to TrOCR with no line
    segmentation at all, to quantify (not just qualitatively assert) how
    much line segmentation actually helps."""
    processor, model = service._get_model(lang)
    device = service._resolve_device()
    import torch

    t0 = time.perf_counter()
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    eos = model.config.eos_token_id or model.generation_config.eos_token_id
    pad = model.config.pad_token_id or model.generation_config.pad_token_id
    with torch.inference_mode():
        generated_ids = model.generate(pixel_values, max_new_tokens=256, eos_token_id=eos, pad_token_id=pad, use_cache=True)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    elapsed = time.perf_counter() - t0
    return text, elapsed


def main():
    scratch = os.environ.get(
        "OCR_EVAL_SCRATCH",
        str(Path.home() / "AppData/Local/Temp/claude/c--Graduation-grad-chatbot-Ibrahim-Hybrid"
            "/0106ed7c-ffd5-4303-8301-6f4dff860cbc/scratchpad/ocr_eval"),
    )
    manifest_path = os.path.join(scratch, "khatt_manifest.json")

    print("=" * 100)
    print("HANDWRITTEN OCR EVALUATION — Tasks 2 & 6")
    print("=" * 100)

    print("\nLoading real English samples (Teklia/IAM-line, cached)...")
    en_samples = load_real_english_samples(6)
    print(f"  {len(en_samples)} English samples loaded.")

    print("Loading real Arabic samples (KHATT, cached from manifest)...")
    ar_samples = load_real_arabic_samples(manifest_path)
    print(f"  {len(ar_samples)} Arabic samples loaded.")

    font_path = _find_english_handwriting_font()
    print(f"Handwriting-style font found: {font_path or '(none — synthetic sample uses default font)'}")
    mixed_sample = make_mixed_synthetic_sample(font_path)

    en_noisy = make_noisy_variant(en_samples[2])
    ar_noisy = make_noisy_variant(ar_samples[2])

    en_multiline = make_multiline_page(en_samples[:3], "en")
    ar_multiline = make_multiline_page(ar_samples[:3], "ar")

    from services.handwritten_ocr_service import get_handwritten_ocr_service
    service = get_handwritten_ocr_service()

    results = []

    def _eval(sample: Sample, batched: bool = False):
        rss_before = rss_mb()
        if batched:
            text, elapsed, debug = run_batched(service, sample.image, sample.lang if sample.lang != "mixed" else "en")
        else:
            text, elapsed, debug = run_sequential(service, sample.image, sample.lang if sample.lang != "mixed" else "en")
        rss_after = rss_mb()
        c = cer(sample.ground_truth, text)
        w = wer(sample.ground_truth, text)
        row = {
            "id": sample.id, "lang": sample.lang, "kind": sample.kind,
            "mode": "batched" if batched else "sequential",
            "num_lines": debug.get("num_lines", 1),
            "latency_s": round(elapsed, 3),
            "latency_per_line_s": round(elapsed / max(1, debug.get("num_lines", 1)), 3),
            "cer": round(c, 3), "wer": round(w, 3),
            "rss_before_mb": round(rss_before, 1) if rss_before else None,
            "rss_after_mb": round(rss_after, 1) if rss_after else None,
            "gt_preview": sample.ground_truth[:60],
            "hyp_preview": text[:60],
        }
        results.append(row)
        return row

    print("\n--- Loading TrOCR models (one-time, cached) and warming up ---")
    t0 = time.perf_counter()
    service._get_model("en")
    service._get_model("ar")
    print(f"Both models loaded in {time.perf_counter() - t0:.1f}s. RSS after load: {rss_mb():.0f} MB" if rss_mb() else "")

    print("\n--- Single-line samples (sequential, current production code path) ---")
    for s in en_samples + ar_samples + [mixed_sample, en_noisy, ar_noisy]:
        row = _eval(s)
        print(f"  [{row['lang']:5}] {row['id']:20} CER={row['cer']:.3f} WER={row['wer']:.3f} "
              f"latency={row['latency_s']}s  gt={row['gt_preview']!r}  hyp={row['hyp_preview']!r}")

    print("\n--- Multi-line pages: WITH segmentation (current pipeline) vs WITHOUT ---")
    for page in (en_multiline, ar_multiline):
        row = _eval(page)
        print(f"  [segmented]   [{row['lang']}] lines={row['num_lines']} CER={row['cer']:.3f} "
              f"WER={row['wer']:.3f} latency={row['latency_s']}s")
        hyp_ns, el_ns = run_no_segmentation(service, page.image, page.lang)
        c_ns = cer(page.ground_truth, hyp_ns)
        w_ns = wer(page.ground_truth, hyp_ns)
        print(f"  [unsegmented] [{page.lang}] CER={c_ns:.3f} WER={w_ns:.3f} latency={el_ns:.3f}s "
              f"hyp={hyp_ns[:60]!r}")
        results.append({"id": page.id + "-nosegmentation", "lang": page.lang, "kind": "real",
                         "mode": "no_segmentation", "num_lines": 1, "latency_s": round(el_ns, 3),
                         "latency_per_line_s": round(el_ns, 3), "cer": round(c_ns, 3), "wer": round(w_ns, 3),
                         "gt_preview": page.ground_truth[:60], "hyp_preview": hyp_ns[:60]})

    print("\n--- Batching benchmark: sequential vs batched on multi-line pages ---")
    for page in (en_multiline, ar_multiline):
        seq = next(r for r in results if r["id"] == page.id and r["mode"] == "sequential")
        rss_before = rss_mb()
        batched_row = _eval(page, batched=True)
        print(f"  [{page.lang}] sequential: {seq['latency_s']}s ({seq['latency_per_line_s']}s/line) "
              f"CER={seq['cer']:.3f}  |  batched: {batched_row['latency_s']}s "
              f"({batched_row['latency_per_line_s']}s/line) CER={batched_row['cer']:.3f}  "
              f"RSS before={batched_row['rss_before_mb']}MB after={batched_row['rss_after_mb']}MB")

    print("\n--- Lightweight alternative: trocr-small-handwritten vs trocr-base-handwritten (English) ---")
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        import torch

        small_name = "microsoft/trocr-small-handwritten"
        t0 = time.perf_counter()
        small_processor = TrOCRProcessor.from_pretrained(small_name)
        small_model = VisionEncoderDecoderModel.from_pretrained(small_name)
        small_model.eval()
        load_time = time.perf_counter() - t0
        print(f"  trocr-small loaded in {load_time:.1f}s")

        small_rows = []
        for s in en_samples:
            buf = io.BytesIO()
            s.image.save(buf, format="PNG")
            t0 = time.perf_counter()
            pv = small_processor(images=s.image, return_tensors="pt").pixel_values
            with torch.inference_mode():
                ids = small_model.generate(pv, max_new_tokens=256, use_cache=True)
            text = small_processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            elapsed = time.perf_counter() - t0
            c, w = cer(s.ground_truth, text), wer(s.ground_truth, text)
            small_rows.append({"id": s.id, "cer": round(c, 3), "wer": round(w, 3), "latency_s": round(elapsed, 3)})
            print(f"    [small] {s.id:16} CER={c:.3f} WER={w:.3f} latency={elapsed:.3f}s")

        base_rows = [r for r in results if r["lang"] == "en" and r["kind"] == "real" and r["mode"] == "sequential" and "multiline" not in r["id"] and "noisy" not in r["id"]]
        avg_small_cer = sum(r["cer"] for r in small_rows) / len(small_rows)
        avg_base_cer = sum(r["cer"] for r in base_rows) / len(base_rows)
        avg_small_lat = sum(r["latency_s"] for r in small_rows) / len(small_rows)
        avg_base_lat = sum(r["latency_s"] for r in base_rows) / len(base_rows)
        print(f"\n  AVERAGE — base: CER={avg_base_cer:.3f} latency={avg_base_lat:.3f}s/line  |  "
              f"small: CER={avg_small_cer:.3f} latency={avg_small_lat:.3f}s/line")
        results.append({"comparison": "trocr-small vs trocr-base", "small_avg_cer": round(avg_small_cer, 3),
                         "base_avg_cer": round(avg_base_cer, 3), "small_avg_latency_s": round(avg_small_lat, 3),
                         "base_avg_latency_s": round(avg_base_lat, 3)})
    except Exception as e:
        print(f"  SKIPPED — could not load/benchmark trocr-small: {e}")

    out_path = os.path.join(scratch, "ocr_eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
