"""
Diagnostic: is services.handwritten_ocr_service._preprocess() (designed for
photographed pages — resize bounds + autocontrast) actively hurting already
-clean, pre-cropped benchmark images (IAM/KHATT), or is the high observed
CER simply this TrOCR checkpoint's real accuracy ceiling on this data?
Compares: raw image -> TrOCRProcessor directly, vs raw image -> app's
_preprocess() -> TrOCRProcessor.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from scripts.evaluate_handwritten_ocr import cer, load_real_english_samples


def main():
    import torch
    from services.handwritten_ocr_service import _preprocess, get_handwritten_ocr_service

    service = get_handwritten_ocr_service()
    processor, model = service._get_model("en")

    def _generate(img):
        pv = processor(images=img, return_tensors="pt").pixel_values
        eos = model.config.eos_token_id or model.generation_config.eos_token_id
        pad = model.config.pad_token_id or model.generation_config.pad_token_id
        with torch.inference_mode():
            ids = model.generate(pv, max_new_tokens=256, eos_token_id=eos, pad_token_id=pad, use_cache=True)
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    samples = load_real_english_samples(6)
    for s in samples:
        raw_text = _generate(s.image)
        buf = io.BytesIO()
        s.image.save(buf, format="PNG")
        prepped_img = _preprocess(buf.getvalue())
        prepped_text = _generate(prepped_img)

        raw_cer = cer(s.ground_truth, raw_text)
        prepped_cer = cer(s.ground_truth, prepped_text)
        print(f"[{s.id}] size={s.image.size}")
        print(f"  gt:      {s.ground_truth!r}")
        print(f"  raw:     CER={raw_cer:.3f}  hyp={raw_text!r}")
        print(f"  prepped: CER={prepped_cer:.3f}  hyp={prepped_text!r}  (resized to {prepped_img.size})")


if __name__ == "__main__":
    main()
