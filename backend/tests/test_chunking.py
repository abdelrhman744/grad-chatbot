"""
Task 1 — semantic chunking bug fix tests.

Root cause under test: `_semantic_split_documents()` in
services/rag_service.py used to compute `chunk_text` per sentence-group and
never append it (or a Document wrapping it) to its `out` list — every
multi-sentence document silently produced zero chunks. These tests exercise
the fixed function directly for single-sentence / multi-sentence / long /
Arabic / English / mixed-language documents, then prove the fix actually
reaches the real code path (not just a stub) with the real local embedding
model, and finally prove `_hybrid_split_documents` (the strategy actually
configured via CHUNKING_STRATEGY=hybrid) is untouched/unaffected.
"""

from __future__ import annotations

import hashlib
from typing import List

import numpy as np
import pytest
from langchain_core.documents import Document

import services.rag_service as rag_service
from config import settings


class FakeEmbeddings:
    """
    Deterministic, dependency-free bag-of-words embedding stub: shares a
    dimension per distinct word (stable hash, not Python's randomized
    `hash()`), so sentences sharing vocabulary land closer together in
    cosine space than unrelated sentences — enough for the breakpoint
    logic in `_semantic_split_documents` to behave meaningfully, without
    downloading/loading the real model for every fast unit test.
    """

    DIM = 128

    def _vec(self, text: str) -> List[float]:
        vec = np.zeros(self.DIM)
        for word in text.lower().split():
            idx = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self.DIM
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec[0] = 1.0
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]


@pytest.fixture
def fake_embeddings(monkeypatch):
    fake = FakeEmbeddings()
    monkeypatch.setattr(rag_service, "embeddings", fake)
    return fake


def _doc(text: str, **meta) -> Document:
    return Document(page_content=text, metadata={"source": "test.txt", "file_type": "txt", "page": 0, **meta})


EN_MULTI = (
    "The mitochondria is the powerhouse of the cell. It generates most of the cell's ATP. "
    "Photosynthesis occurs in the chloroplast. Plants use sunlight to produce glucose. "
    "The stock market closed higher today. Tech shares led the rally. "
    "A new bridge was opened downtown. Traffic patterns will change next week."
)

AR_MULTI = (
    "الخلية هي الوحدة الأساسية للحياة. تحتوي الخلية على نواة ومكونات مختلفة. "
    "التمثيل الضوئي يحدث في البلاستيدات الخضراء. تستخدم النباتات ضوء الشمس لإنتاج الغذاء. "
    "ارتفع سوق الأسهم اليوم بشكل ملحوظ. قادت أسهم التكنولوجيا هذا الارتفاع."
)

MIXED_AR_EN = (
    "عايز أعرف الـ credit hours بتاعة مادة Computer Science. "
    "The course covers data structures and algorithms. "
    "الامتحان النهائي هيكون في شهر يناير. "
    "Attendance is mandatory for all lab sessions."
)


class TestSemanticChunkingFix:
    def test_single_sentence_document_passthrough(self, fake_embeddings):
        doc = _doc("Just one sentence with no terminator issues here")
        out = rag_service._semantic_split_documents([doc])
        assert len(out) == 1
        assert out[0].page_content == doc.page_content
        assert out[0].metadata["source"] == "test.txt"

    def test_multi_sentence_english_produces_nonempty_chunks(self, fake_embeddings):
        doc = _doc(EN_MULTI)
        out = rag_service._semantic_split_documents([doc])
        assert len(out) > 0, "multi-sentence English document must not silently vanish"
        assert all(c.page_content.strip() for c in out)
        # every chunk must preserve the source document's metadata
        assert all(c.metadata.get("source") == "test.txt" for c in out)

    def test_multi_sentence_arabic_produces_nonempty_chunks(self, fake_embeddings):
        doc = _doc(AR_MULTI)
        out = rag_service._semantic_split_documents([doc])
        assert len(out) > 0, "multi-sentence Arabic document must not silently vanish"
        assert all(c.page_content.strip() for c in out)

    def test_mixed_arabic_english_produces_nonempty_chunks(self, fake_embeddings):
        doc = _doc(MIXED_AR_EN)
        out = rag_service._semantic_split_documents([doc])
        assert len(out) > 0, "mixed AR/EN document must not silently vanish"
        assert all(c.page_content.strip() for c in out)

    def test_long_document_respects_max_chars_safety_cap(self, fake_embeddings, monkeypatch):
        # Force (near-)zero breakpoints so the whole document collapses into
        # ONE oversized group, forcing the SEMANTIC_CHUNK_MAX_CHARS fallback
        # splitter path (previously dead code) to actually run.
        monkeypatch.setattr(settings, "SEMANTIC_CHUNK_BREAKPOINT_PERCENTILE", 100.0)
        long_text = ". ".join([f"This is filler sentence number {i} about a shared topic" for i in range(200)]) + "."
        doc = _doc(long_text)
        out = rag_service._semantic_split_documents([doc])
        assert len(out) > 1, "an oversized single group must be split by the fallback splitter"
        max_chars = settings.SEMANTIC_CHUNK_MAX_CHARS
        assert all(len(c.page_content) <= max_chars for c in out), (
            "no emitted chunk may exceed SEMANTIC_CHUNK_MAX_CHARS"
        )

    def test_chunks_do_not_share_metadata_dict(self, fake_embeddings):
        doc = _doc(EN_MULTI)
        out = rag_service._semantic_split_documents([doc])
        assert len(out) >= 2
        out[0].metadata["chunk_index"] = 0
        out[1].metadata["chunk_index"] = 1
        assert out[0].metadata["chunk_index"] != out[1].metadata["chunk_index"] or (
            out[0].metadata is not out[1].metadata
        )
        assert out[0].metadata is not out[1].metadata, "each chunk must own an independent metadata dict"

    def test_empty_document_produces_no_chunks_without_crashing(self, fake_embeddings):
        doc = _doc("   ")
        out = rag_service._semantic_split_documents([doc])
        assert out == []

    def test_update_db_files_falls_back_on_empty_semantic_output(self, fake_embeddings, monkeypatch):
        """
        _chunk_documents() must never let a strategy that produced zero
        chunks from non-empty input pass through silently — it must fall
        back to the recursive splitter instead.
        """
        monkeypatch.setattr(settings, "CHUNKING_STRATEGY", "semantic")
        monkeypatch.setattr(rag_service, "_semantic_split_documents", lambda docs: [])
        doc = _doc(EN_MULTI)
        chunks = rag_service._chunk_documents([doc])
        assert len(chunks) > 0, "empty semantic output must fall back to recursive chunking, not vanish"


class TestHybridUnaffectedByFix:
    """Task 1's edit only touches _semantic_split_documents; hybrid (the
    configured CHUNKING_STRATEGY) must be provably unaffected."""

    def test_hybrid_still_produces_chunks(self, fake_embeddings):
        doc = _doc(EN_MULTI)
        out = rag_service._hybrid_split_documents([doc])
        assert len(out) > 0
        assert all(c.page_content.strip() for c in out)

    def test_hybrid_never_calls_semantic_function(self, fake_embeddings, monkeypatch):
        called = {"hit": False}

        def _poison(*args, **kwargs):
            called["hit"] = True
            raise AssertionError("hybrid must never call _semantic_split_documents")

        monkeypatch.setattr(rag_service, "_semantic_split_documents", _poison)
        doc = _doc(EN_MULTI)
        rag_service._hybrid_split_documents([doc])
        assert called["hit"] is False


@pytest.mark.real_model
class TestSemanticChunkingRealEmbeddingModel:
    """
    Not a stub — uses the actual local sentence-transformers model via
    services.embeddings_provider.get_embeddings() (already cached on this
    machine, so no network needed) to prove the fix works on the real code
    path, not just against FakeEmbeddings.
    """

    def test_real_model_produces_chunks_for_mixed_document(self, monkeypatch):
        from services.embeddings_provider import get_embeddings

        real = get_embeddings()
        monkeypatch.setattr(rag_service, "embeddings", real)
        doc = _doc(MIXED_AR_EN)
        out = rag_service._semantic_split_documents([doc])
        assert len(out) > 0, "real embedding model must also produce non-empty semantic chunks"
        assert all(c.page_content.strip() for c in out)


@pytest.mark.live_qdrant
class TestLiveIngestionPipelines:
    """
    Pipelines A & B from the task spec, run against a real Qdrant (and
    MinIO) via `docker compose up -d qdrant minio` — full
    document -> chunk -> embed -> Qdrant -> retrieve, no mocking of the
    vector store. Each test uses its own fresh conversation_id and cleans
    up after itself via delete_conversation_documents.
    """

    def _conv_id(self) -> str:
        import uuid

        return f"pytest-pipeline-{uuid.uuid4().hex[:12]}"

    def _run_pipeline(self, monkeypatch, strategy: str, unique_phrase: str, body: str):
        monkeypatch.setattr(settings, "CHUNKING_STRATEGY", strategy)
        conv_id = self._conv_id()
        try:
            chunks_added = rag_service.update_db_files(
                [{"filename": f"pipeline_{strategy}.txt", "data": body.encode("utf-8")}],
                conversation_id=conv_id,
            )
            assert chunks_added > 0, f"{strategy} ingestion must produce at least one chunk"

            results = rag_service.retrieve(unique_phrase, conversation_id=conv_id, top_k=5)
            assert len(results) > 0, f"{strategy} pipeline: unique phrase must be retrievable"
            assert any(unique_phrase in r["text"] for r in results)
        finally:
            rag_service.delete_conversation_documents(conv_id)

    def test_pipeline_a_hybrid_ingest_and_retrieve(self, monkeypatch):
        body = (
            "PIPELINE-A-HYBRID-MARKER-4471 is the unique identifier for this test document. "
            + EN_MULTI
        )
        self._run_pipeline(monkeypatch, "hybrid", "PIPELINE-A-HYBRID-MARKER-4471", body)

    def test_pipeline_b_semantic_ingest_and_retrieve(self, monkeypatch):
        body = (
            "PIPELINE-B-SEMANTIC-MARKER-8823 is the unique identifier for this test document. "
            + EN_MULTI
        )
        self._run_pipeline(monkeypatch, "semantic", "PIPELINE-B-SEMANTIC-MARKER-8823", body)

    def test_pipeline_b_semantic_arabic_ingest_and_retrieve(self, monkeypatch):
        body = "المعرف-الفريد-٩٩٨٨ هو رقم تعريف هذا المستند التجريبي. " + AR_MULTI
        self._run_pipeline(monkeypatch, "semantic", "المعرف-الفريد-٩٩٨٨", body)
