"""
Follow-up experiments for Task 6, run after evaluate_handwritten_ocr.py's
initial results raised two open questions:
  1. The first batching benchmark used only 3-line pages — too small a
     sample to draw a production conclusion (Arabic got SLOWER when
     batched at that size). Re-run at a more realistic page size (6-8
     lines) to see whether batching's overhead amortizes better.
  2. Both models currently decode greedily (no `num_beams` passed to
     `generate()`, defaulting to beam=1). Published TrOCR benchmarks
     typically use beam search. Check whether beam search meaningfully
     improves accuracy at an acceptable latency cost on CPU.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
from PIL import Image

from scripts.evaluate_handwritten_ocr import (
    cer, wer, load_real_english_samples, load_real_arabic_samples,
    make_multiline_page, rss_mb,
)

SCRATCH = os.environ.get(
    "OCR_EVAL_SCRATCH",
    str(Path.home() / "AppData/Local/Temp/claude/c--Graduation-grad-chatbot-Ibrahim-Hybrid"
        "/0106ed7c-ffd5-4303-8301-6f4dff860cbc/scratchpad/ocr_eval"),
)


def run_batched_bounded(service, image, lang, max_batch=16):
    import torch
    from services.handwritten_ocr_service import _preprocess, _segment_lines

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    prepped = _preprocess(buf.getvalue())
    lines = _segment_lines(prepped)

    processor, model = service._get_model(lang)
    device = service._resolve_device()
    texts = []
    t0 = time.perf_counter()
    for i in range(0, len(lines), max_batch):
        batch = lines[i:i + max_batch]
        pixel_values = processor(images=batch, return_tensors="pt").pixel_values.to(device)
        eos = model.config.eos_token_id or model.generation_config.eos_token_id
        pad = model.config.pad_token_id or model.generation_config.pad_token_id
        with torch.inference_mode():
            ids = model.generate(pixel_values, max_new_tokens=256, eos_token_id=eos, pad_token_id=pad, use_cache=True)
        texts.extend(t.strip() for t in processor.batch_decode(ids, skip_special_tokens=True))
    elapsed = time.perf_counter() - t0
    return "\n".join(t for t in texts if t), elapsed, len(lines)


def run_sequential_timed(service, image, lang):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    t0 = time.perf_counter()
    text, debug = service.recognize_with_debug(buf.getvalue(), lang)
    elapsed = time.perf_counter() - t0
    return text, elapsed, debug["num_lines"]


def run_with_beams(service, image_bytes, lang, num_beams):
    import torch
    from services.handwritten_ocr_service import _preprocess, _segment_lines

    prepped = _preprocess(image_bytes)
    lines = _segment_lines(prepped)
    processor, model = service._get_model(lang)
    device = service._resolve_device()
    texts = []
    t0 = time.perf_counter()
    for img in lines:
        pv = processor(images=img, return_tensors="pt").pixel_values.to(device)
        eos = model.config.eos_token_id or model.generation_config.eos_token_id
        pad = model.config.pad_token_id or model.generation_config.pad_token_id
        with torch.inference_mode():
            ids = model.generate(pv, max_new_tokens=256, eos_token_id=eos, pad_token_id=pad,
                                  use_cache=True, num_beams=num_beams)
        texts.append(processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
    elapsed = time.perf_counter() - t0
    return "\n".join(t for t in texts if t), elapsed


def main():
    from services.handwritten_ocr_service import get_handwritten_ocr_service
    service = get_handwritten_ocr_service()
    service._get_model("en")
    service._get_model("ar")

    print("=" * 100)
    print("FOLLOW-UP 1: batching at realistic page scale (6-8 lines)")
    print("=" * 100)

    en_samples = load_real_english_samples(6)
    ar_samples = load_real_arabic_samples(os.path.join(SCRATCH, "khatt_manifest.json"))

    en_page = make_multiline_page(en_samples, "en")   # 6 lines
    ar_page = make_multiline_page(ar_samples, "ar")   # 8 lines

    for page in (en_page, ar_page):
        seq_text, seq_time, seq_lines = run_sequential_timed(service, page.image, page.lang)
        rss_before = rss_mb()
        batch_text, batch_time, batch_lines = run_batched_bounded(service, page.image, page.lang)
        rss_after = rss_mb()
        seq_cer = cer(page.ground_truth, seq_text)
        batch_cer = cer(page.ground_truth, batch_text)
        print(f"\n[{page.lang}] {seq_lines} lines")
        print(f"  sequential: {seq_time:.3f}s ({seq_time/seq_lines:.3f}s/line) CER={seq_cer:.3f}")
        print(f"  batched:    {batch_time:.3f}s ({batch_time/batch_lines:.3f}s/line) CER={batch_cer:.3f} "
              f"speedup={seq_time/batch_time:.2f}x  RSS {rss_before:.0f}->{rss_after:.0f}MB")

    print("\n" + "=" * 100)
    print("FOLLOW-UP 2: greedy (current) vs beam search (num_beams=4) on the worst-performing samples")
    print("=" * 100)

    worst_en = [s for s in en_samples if s.id == "iam-line-1490"][0]  # CER 0.722 in first run
    worst_ar = ar_samples[1]  # "test/AHTD3A0002_Para4_2" -> hyp was just "."

    for sample in (worst_en, worst_ar):
        buf = io.BytesIO()
        sample.image.save(buf, format="PNG")
        data = buf.getvalue()
        t0 = time.perf_counter()
        greedy_text = service.recognize(data, sample.lang)
        greedy_time = time.perf_counter() - t0
        greedy_cer = cer(sample.ground_truth, greedy_text)

        beam_text, beam_time = run_with_beams(service, data, sample.lang, num_beams=4)
        beam_cer = cer(sample.ground_truth, beam_text)

        print(f"\n[{sample.lang}] {sample.id}")
        print(f"  gt:     {sample.ground_truth!r}")
        print(f"  greedy: CER={greedy_cer:.3f} latency={greedy_time:.3f}s  hyp={greedy_text!r}")
        print(f"  beam=4: CER={beam_cer:.3f} latency={beam_time:.3f}s  hyp={beam_text!r}")


if __name__ == "__main__":
    main()
