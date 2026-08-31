"""
Tasks 2 & 6 — functional (correctness, not benchmark) tests for handwritten
OCR. Benchmarking/CER-WER evidence lives in
scripts/evaluate_handwritten_ocr.py; this file proves the real code paths
behave correctly: real models, real (downloaded, cached) sample images —
no mocking of TrOCR itself.

Marked `real_model` (slow — loads two ~334M-parameter TrOCR models on
first run) and `live_qdrant` for the Pipeline E integration test.
"""

from __future__ import annotations

import io
import json
import os
import uuid
from pathlib import Path

import pytest
from PIL import Image

from services.handwritten_ocr_service import get_handwritten_ocr_service

SCRATCH = os.environ.get(
    "OCR_EVAL_SCRATCH",
    str(Path.home() / "AppData/Local/Temp/claude/c--Graduation-grad-chatbot-Ibrahim-Hybrid"
        "/0106ed7c-ffd5-4303-8301-6f4dff860cbc/scratchpad/ocr_eval"),
)
_KHATT_MANIFEST = os.path.join(SCRATCH, "khatt_manifest.json")


def _khatt_sample(index: int = 0):
    with open(_KHATT_MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    entry = manifest[index]
    return Image.open(entry["local_path"]).convert("RGB"), entry["text"]


def _iam_sample(index: int = 0):
    from datasets import load_dataset

    ds = load_dataset("Teklia/IAM-line", split="test")
    ex = ds[index]
    return ex["image"].convert("RGB"), ex["text"].strip()


def _to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.real_model
class TestHandwrittenOCRCorrectness:
    def test_english_real_handwriting_returns_nonempty_plausible_text(self):
        img, gt = _iam_sample(0)
        service = get_handwritten_ocr_service()
        text = service.recognize(_to_bytes(img), "en")
        assert text.strip(), "OCR must return non-empty text for a real, legible handwriting line"
        # loose plausibility check: at least one ground-truth word recognizable
        gt_words = {w.strip(".,;:!?").lower() for w in gt.split() if len(w) > 3}
        hyp_words = {w.strip(".,;:!?").lower() for w in text.split() if len(w) > 3}
        assert gt_words & hyp_words, f"expected some word overlap; gt={gt!r} hyp={text!r}"

    def test_arabic_real_handwriting_returns_nonempty_text(self):
        img, gt = _khatt_sample(0)
        service = get_handwritten_ocr_service()
        text = service.recognize(_to_bytes(img), "ar")
        assert text.strip(), "OCR must return non-empty text for a real Arabic handwriting line"

    def test_unsupported_language_rejected(self):
        img, _ = _iam_sample(0)
        service = get_handwritten_ocr_service()
        with pytest.raises(Exception):
            service.recognize(_to_bytes(img), "fr")

    def test_invalid_image_bytes_rejected(self):
        from services.handwritten_ocr_service import InvalidImageError

        service = get_handwritten_ocr_service()
        with pytest.raises(InvalidImageError):
            service.recognize(b"not an image", "en")

    def test_multiline_page_segments_into_multiple_lines(self):
        img1, t1 = _iam_sample(0)
        img2, t2 = _iam_sample(1)
        width = max(img1.width, img2.width) + 40
        height = img1.height + img2.height + 90
        page = Image.new("RGB", (width, height), "white")
        page.paste(img1, (20, 30))
        page.paste(img2, (20, img1.height + 60))

        service = get_handwritten_ocr_service()
        text, debug = service.recognize_with_debug(_to_bytes(page), "en")
        assert debug["num_lines"] >= 2, "a real two-line page must be segmented into >= 2 lines"
        assert text.strip()


@pytest.mark.real_model
class TestBatchedInference:
    """Task 6 — batching (services.handwritten_ocr_service._recognize_lines
    / _recognize_batch) must produce the same text as the previous strictly
    sequential per-line loop, on real (not mocked) TrOCR inference."""

    def _two_real_lines(self, lang: str):
        if lang == "en":
            img1, _ = _iam_sample(0)
            img2, _ = _iam_sample(1)
        else:
            img1, _ = _khatt_sample(0)
            img2, _ = _khatt_sample(1)
        return [img1, img2]

    def test_batched_matches_sequential_english(self):
        service = get_handwritten_ocr_service()
        images = self._two_real_lines("en")

        sequential = [service._recognize_image(img, "en") for img in images]
        batched = service._recognize_batch(images, "en")

        assert batched == sequential, (
            "batched inference must produce IDENTICAL text to sequential "
            "inference for the same input images"
        )

    def test_batched_matches_sequential_arabic(self):
        service = get_handwritten_ocr_service()
        images = self._two_real_lines("ar")

        sequential = [service._recognize_image(img, "ar") for img in images]
        batched = service._recognize_batch(images, "ar")

        assert batched == sequential

    def test_recognize_lines_single_image_skips_batching(self):
        """A single line must go through the plain per-image path — there
        is nothing to batch, and _recognize_batch must not be called."""
        service = get_handwritten_ocr_service()
        img, _ = _iam_sample(0)

        called = {"hit": False}
        real_batch = service._recognize_batch

        def _spy(*args, **kwargs):
            called["hit"] = True
            return real_batch(*args, **kwargs)

        service._recognize_batch = _spy
        try:
            texts = service._recognize_lines([img], "en")
        finally:
            service._recognize_batch = real_batch

        assert len(texts) == 1
        assert called["hit"] is False

    def test_recognize_lines_respects_max_batch_size(self, monkeypatch):
        """More lines than HANDWRITTEN_OCR_MAX_BATCH_SIZE must be split
        into multiple bounded sub-batch calls, never one unbounded batch."""
        from config import settings

        monkeypatch.setattr(settings, "HANDWRITTEN_OCR_MAX_BATCH_SIZE", 2)
        service = get_handwritten_ocr_service()
        img, _ = _iam_sample(0)
        images = [img] * 5  # 5 identical images -> ceil(5/2) = 3 sub-batches

        seen_batch_sizes = []
        real_batch = service._recognize_batch

        def _spy(batch, lang):
            seen_batch_sizes.append(len(batch))
            return real_batch(batch, lang)

        service._recognize_batch = _spy
        try:
            texts = service._recognize_lines(images, "en")
        finally:
            service._recognize_batch = real_batch

        assert len(texts) == 5
        assert seen_batch_sizes == [2, 2, 1]
        assert all(size <= 2 for size in seen_batch_sizes)

    def test_batch_failure_falls_back_to_sequential(self, monkeypatch):
        """If a batched forward pass raises, that sub-batch must still be
        recognized (sequentially), not silently dropped."""
        service = get_handwritten_ocr_service()
        img, _ = _iam_sample(0)

        def _broken_batch(images, lang):
            raise RuntimeError("simulated batch failure")

        monkeypatch.setattr(service, "_recognize_batch", _broken_batch)
        texts = service._recognize_lines([img, img], "en")
        assert len(texts) == 2
        assert all(t.strip() for t in texts), "fallback must still produce real recognized text"


@pytest.mark.real_model
@pytest.mark.live_qdrant
class TestPipelineE_HandwrittenOCRToRetrieval:
    """Handwritten image -> preprocess -> line segmentation -> OCR -> text
    -> chunk -> embed -> Qdrant -> retrieval, end to end, against real
    Qdrant (docker compose up -d qdrant minio)."""

    def test_ocr_text_is_indexed_and_retrievable(self):
        import services.rag_service as rag_service

        img, gt = _iam_sample(3)
        service = get_handwritten_ocr_service()
        text = service.recognize(_to_bytes(img), "en")
        assert text.strip()

        conv_id = f"pytest-ocr-pipeline-{uuid.uuid4().hex[:8]}"
        try:
            added = rag_service.update_db_files(
                [{"filename": "handwritten_sample.handwritten-ocr.txt", "data": text.encode("utf-8")}],
                conversation_id=conv_id,
            )
            assert added > 0, "OCR'd text must produce at least one indexed chunk"

            # Query using a distinctive word actually present in what the
            # model recognized (not the ground truth, since OCR isn't
            # perfect — this proves the REAL recognized text is retrievable).
            words = [w.strip(".,;:!?\"'").lower() for w in text.split() if len(w) > 4]
            assert words, "recognized text should contain at least one longer word to query with"
            query_word = words[0]
            results = rag_service.retrieve(query_word, conversation_id=conv_id, top_k=5)
            assert len(results) > 0, "the indexed OCR text must be retrievable by its own content"
        finally:
            rag_service.delete_conversation_documents(conv_id)
