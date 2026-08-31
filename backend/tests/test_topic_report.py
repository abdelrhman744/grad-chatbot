"""
Tests for services.report_service's topic-scoped report pipeline
(_topic_digest, _topic_facts, _reduce_topic_report, _fallback_topic_sections,
render_topic_report_pdf) — the adaptive, concise, page-number-free report
structure requested for topic-based reports, as opposed to the unchanged
whole-document "detailed analysis" pipeline (build_report_data /
render_report_pdf).
"""

import json
import sys
import types

import pytest

# services.report_service imports services.rag_service, which in this
# sandboxed test environment pulls in sentence-transformers/torch (not
# installed here, and unrelated to what this test file exercises — it
# only needs detect_language/get_document_pages/retrieve to exist as
# importable names). Stub it out before importing report_service, same
# technique used in test_docx_ocr.py for services.ocr_service.
if "services.rag_service" not in sys.modules:
    _fake_rag = types.ModuleType("services.rag_service")
    _fake_rag.detect_language = lambda text: "ar" if any("\u0600" <= c <= "\u06FF" for c in (text or "")) else "en"
    _fake_rag.get_document_pages = lambda filename: []
    _fake_rag.retrieve = lambda *a, **k: []
    sys.modules["services.rag_service"] = _fake_rag

import services.report_service as report_service


def _map_result(title, summary, key_points=None, technical_terms=None):
    return {
        "section_title": title,
        "summary": summary,
        "key_points": key_points or [],
        "definitions": [],
        "technical_terms": technical_terms or [],
        "numbers": [],
        "equations": [],
        "best_practices": [],
        "figures_tables": [],
    }


def test_topic_digest_has_no_page_or_rank_labels():
    results = [
        _map_result("AI in healthcare", "AI helps diagnose disease faster."),
        _map_result("Applications", "Used in imaging and drug discovery."),
    ]
    digest = report_service._topic_digest(results)
    assert "p." not in digest.lower()
    assert "page" not in digest.lower()
    assert "AI helps diagnose disease faster." in digest


def test_topic_facts_dedupes_and_has_no_page_labels():
    results = [
        _map_result("A", "s", key_points=["Faster diagnosis"], technical_terms=["RAG"]),
        _map_result("B", "s", key_points=["Faster diagnosis"], technical_terms=["RAG", "LLM"]),
    ]
    facts = report_service._topic_facts(results)
    assert facts.count("Faster diagnosis") == 1
    assert facts.count("RAG") == 1
    assert "LLM" in facts
    assert "(p." not in facts


def test_fallback_topic_sections_empty_digest():
    assert report_service._fallback_topic_sections("", "en") == {"sections": []}


def test_fallback_topic_sections_uses_digest_when_llm_unavailable():
    result = report_service._fallback_topic_sections("Some grounded digest text.", "en")
    assert result["sections"]
    assert "Some grounded digest text." in result["sections"][0]["body"]


def test_reduce_topic_report_parses_structured_llm_output(monkeypatch):
    payload = {
        "sections": [
            {"heading": "Introduction", "body": "AI is transforming healthcare.", "bullets": []},
            {"heading": "Applications", "body": "", "bullets": ["Diagnostics", "Drug discovery"]},
            {"heading": "Conclusion", "body": "AI will keep growing in healthcare.", "bullets": []},
        ]
    }
    monkeypatch.setattr(
        report_service._topic_reduce_llm, "invoke", lambda prompt: json.dumps(payload)
    )
    result = report_service._reduce_topic_report(
        "AI helps diagnose disease.", "- Diagnostics\n- Drug discovery", "AI in healthcare", "en"
    )
    headings = [s["heading"] for s in result["sections"]]
    assert headings == ["Introduction", "Applications", "Conclusion"]
    assert result["sections"][1]["bullets"] == ["Diagnostics", "Drug discovery"]


def test_reduce_topic_report_falls_back_on_malformed_json(monkeypatch):
    monkeypatch.setattr(
        report_service._topic_reduce_llm, "invoke", lambda prompt: "not json at all"
    )
    result = report_service._reduce_topic_report(
        "AI helps diagnose disease.", "- Diagnostics", "AI in healthcare", "en"
    )
    assert result["sections"]  # falls back to digest-based section, not empty
    assert "AI helps diagnose disease." in result["sections"][0]["body"]


def test_reduce_topic_report_empty_input_returns_no_sections():
    result = report_service._reduce_topic_report("", "", "Nonexistent topic", "en")
    assert result == {"sections": []}


def test_render_topic_report_pdf_produces_pdf_bytes_without_page_clutter():
    data = {
        "topic": "الذكاء الاصطناعي في الرعاية الصحية",
        "filename": "الذكاء الاصطناعي في الرعاية الصحية",
        "language": "ar",
        "sources": ["AI Healthcare.pdf", "Machine Learning.docx"],
        "sections": [
            {"heading": "مقدمة", "body": "الذكاء الاصطناعي يغير قطاع الرعاية الصحية.", "bullets": []},
            {"heading": "أهم التطبيقات", "body": "", "bullets": ["التشخيص", "اكتشاف الأدوية"]},
            {"heading": "الخلاصة", "body": "من المتوقع أن يستمر هذا النمو.", "bullets": []},
        ],
    }
    pdf_bytes = report_service.render_topic_report_pdf(data)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 500


def test_render_topic_report_pdf_english_with_no_sources():
    data = {
        "topic": "Machine Learning",
        "language": "en",
        "sources": [],
        "sections": [{"heading": "Introduction", "body": "Machine learning enables systems to learn from data.", "bullets": []}],
    }
    pdf_bytes = report_service.render_topic_report_pdf(data)
    assert pdf_bytes.startswith(b"%PDF")
