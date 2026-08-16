"""
Tasks 3 & 4 — document re-indexing and deletion, run live against real
Qdrant + MinIO (`docker compose up -d qdrant minio`). No mocking of the
vector store: these prove the actual safe-lifecycle invariants — no
stale/duplicate vectors after a re-index, no cross-document or
cross-conversation leakage on delete, idempotent repeated delete.
"""

from __future__ import annotations

import uuid

import pytest

import services.rag_service as rag_service
from services.db_service import get_client, get_collection_name


def _conv_id() -> str:
    return f"pytest-lifecycle-{uuid.uuid4().hex[:12]}"


def _index_one(conv_id: str, filename: str, body: str) -> str:
    """Index one file, return its document_id."""
    added = rag_service.update_db_files(
        [{"filename": filename, "data": body.encode("utf-8")}], conversation_id=conv_id
    )
    assert added > 0
    files = rag_service.list_stored_files(conv_id)
    match = next(f for f in files if f["filename"] == filename)
    return match["document_id"]


def _phrase_found(conv_id: str, phrase: str) -> bool:
    results = rag_service.retrieve(phrase, conversation_id=conv_id, top_k=8)
    return any(phrase in r["text"] for r in results)


def _qdrant_point_count(document_id: str, conv_id: str) -> int:
    doc_filter = rag_service._document_filter(document_id, conv_id)
    return get_client().count(
        collection_name=get_collection_name(), count_filter=doc_filter, exact=True
    ).count


@pytest.fixture
def conv_id():
    cid = _conv_id()
    yield cid
    rag_service.delete_conversation_documents(cid)


@pytest.mark.live_qdrant
class TestReindexing:
    def test_index_then_retrieve(self, conv_id):
        body = "MARKER-ALPHA-9001 describes the Q1 revenue for the WidgetCo division."
        _index_one(conv_id, "doc_a.txt", body)
        assert _phrase_found(conv_id, "MARKER-ALPHA-9001")

    def test_reindex_replaces_content_no_stale_duplicates(self, conv_id):
        v1 = "MARKER-ALPHA-9001 describes the Q1 revenue for the WidgetCo division in January."
        v2 = "MARKER-BETA-7002 describes the corrected Q2 revenue for the WidgetCo division, superseding the prior draft."

        doc_id = _index_one(conv_id, "doc_a.txt", v1)
        assert _phrase_found(conv_id, "MARKER-ALPHA-9001")

        new_chunks = rag_service.reindex_document(
            doc_id, conv_id, new_filename="doc_a.txt", new_data=v2.encode("utf-8")
        )
        assert new_chunks > 0

        # Old content must be gone, new content must be present.
        assert not _phrase_found(conv_id, "MARKER-ALPHA-9001"), (
            "stale v1 content must not remain retrievable after reindex"
        )
        assert _phrase_found(conv_id, "MARKER-BETA-7002")

        # Ground truth: Qdrant's own point count for this document_id must
        # exactly equal the freshly reported chunk count — proves no
        # leftover v1 points coexist with the v2 points.
        assert _qdrant_point_count(doc_id, conv_id) == new_chunks

    def test_reindex_same_document_id_preserved_across_versions(self, conv_id):
        v1 = "MARKER-GAMMA-1 first version of this document."
        v2 = "MARKER-GAMMA-2 second version of this document."
        doc_id = _index_one(conv_id, "doc_c.txt", v1)
        rag_service.reindex_document(doc_id, conv_id, new_filename="doc_c.txt", new_data=v2.encode("utf-8"))
        files = rag_service.list_stored_files(conv_id)
        assert any(f["document_id"] == doc_id for f in files), (
            "document_id must remain the stable identifier across a reindex"
        )

    def test_reindex_unknown_document_raises(self, conv_id):
        with pytest.raises(ValueError):
            rag_service.reindex_document("does-not-exist", conv_id)

    def test_reindex_wrong_conversation_raises(self, conv_id):
        other_conv = _conv_id()
        try:
            v1 = "MARKER-DELTA-3 belongs only to the first conversation."
            doc_id = _index_one(conv_id, "doc_d.txt", v1)
            with pytest.raises(ValueError):
                rag_service.reindex_document(doc_id, other_conv, new_data=b"attempted hijack")
            # original untouched
            assert _phrase_found(conv_id, "MARKER-DELTA-3")
        finally:
            rag_service.delete_conversation_documents(other_conv)


@pytest.mark.live_qdrant
class TestDeletion:
    def test_delete_removes_only_target_document(self, conv_id):
        a_body = "MARKER-DOC-A-5551 is unique to document A."
        b_body = "MARKER-DOC-B-5552 is unique to document B."
        doc_a = _index_one(conv_id, "doc_a.txt", a_body)
        doc_b = _index_one(conv_id, "doc_b.txt", b_body)

        assert _phrase_found(conv_id, "MARKER-DOC-A-5551")
        assert _phrase_found(conv_id, "MARKER-DOC-B-5552")

        removed = rag_service.delete_document(doc_a, conv_id)
        assert removed > 0

        assert not _phrase_found(conv_id, "MARKER-DOC-A-5551"), "deleted document must not appear in results"
        assert _phrase_found(conv_id, "MARKER-DOC-B-5552"), "other document must be unaffected"
        assert _qdrant_point_count(doc_a, conv_id) == 0
        assert _qdrant_point_count(doc_b, conv_id) > 0

    def test_delete_is_idempotent(self, conv_id):
        body = "MARKER-DOC-E-5553 is unique to document E."
        doc_id = _index_one(conv_id, "doc_e.txt", body)

        first = rag_service.delete_document(doc_id, conv_id)
        assert first > 0

        second = rag_service.delete_document(doc_id, conv_id)
        assert second == 0, "repeated delete of an already-deleted document must be a safe no-op"

        third = rag_service.delete_document("never-existed-at-all", conv_id)
        assert third == 0, "deleting a document_id that never existed must also be a safe no-op"

    def test_delete_cannot_cross_conversation_boundary(self, conv_id):
        other_conv = _conv_id()
        try:
            body = "MARKER-DOC-F-5554 belongs only to the real owner conversation."
            doc_id = _index_one(conv_id, "doc_f.txt", body)

            # Attempting to delete it while impersonating a different
            # conversation_id must remove nothing.
            removed = rag_service.delete_document(doc_id, other_conv)
            assert removed == 0

            assert _phrase_found(conv_id, "MARKER-DOC-F-5554"), (
                "document must remain fully intact after a cross-conversation delete attempt"
            )
        finally:
            rag_service.delete_conversation_documents(other_conv)
