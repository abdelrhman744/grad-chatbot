"""
Tests for services.ocr_service._clean_ocr_artifacts and its helpers — the
post-processing step that removes garbled Latin-script noise Tesseract
sometimes hallucinates inside Arabic OCR output (see the function's own
docstring in services/ocr_service.py for the full rationale), while
preserving genuine English words and technical terms.
"""

from services.ocr_service import _clean_ocr_artifacts, _should_keep_latin_token
from config import settings


def test_removes_meaningless_latin_fragment_inside_arabic_line():
    text = "هذا النص يحتوي على IST كلمة زائدة غير موجودة في الأصل"
    cleaned = _clean_ocr_artifacts(text)
    assert "IST" not in cleaned
    assert "هذا النص يحتوي على" in cleaned


def test_preserves_whitelisted_technical_acronyms_in_arabic_line():
    text = "نظام إدارة قواعد البيانات يستخدم تقنية RAG و LLM بشكل واسع"
    cleaned = _clean_ocr_artifacts(text)
    assert "RAG" in cleaned
    assert "LLM" in cleaned


def test_preserves_real_english_words_in_arabic_line():
    text = "الذكاء الاصطناعي أصبح مهما جدا في الرعاية الصحية Healthcare"
    cleaned = _clean_ocr_artifacts(text)
    assert "Healthcare" in cleaned


def test_preserves_camelcase_and_digit_mixed_technical_terms():
    text = "نستخدم FastAPI و Python لبناء API قوي، ونستخدم أيضا GPT4"
    cleaned = _clean_ocr_artifacts(text)
    assert "FastAPI" in cleaned
    assert "Python" in cleaned
    assert "API" in cleaned
    assert "GPT4" in cleaned


def test_does_not_touch_pure_english_text():
    text = "This is a pure English sentence about Machine Learning and AI."
    assert _clean_ocr_artifacts(text) == text


def test_does_not_touch_english_only_documents_at_all():
    """No Arabic anywhere in the text -> the function is a no-op, so an
    entirely English document is guaranteed untouched regardless of what
    any individual token looks like."""
    text = "Zqx Wbrn flon random-looking words that are not real English"
    assert _clean_ocr_artifacts(text) == text


def test_preserves_mixed_document_english_section_untouched():
    """A line with no Arabic characters at all inside an otherwise mixed
    document is left completely alone, even if it contains short/unusual
    tokens — the cleanup only ever touches Arabic-majority lines."""
    text = (
        "هذا القسم باللغة العربية RAG\n"
        "This entire line is English and mentions Zzp as a placeholder token."
    )
    cleaned = _clean_ocr_artifacts(text)
    lines = cleaned.split("\n")
    assert lines[1] == "This entire line is English and mentions Zzp as a placeholder token."


def test_should_keep_latin_token_whitelist():
    assert _should_keep_latin_token("RAG") is True
    assert _should_keep_latin_token("LLM") is True
    assert _should_keep_latin_token("Healthcare") is True


def test_should_keep_latin_token_rejects_short_gibberish():
    assert _should_keep_latin_token("IST") is False


def test_should_keep_latin_token_keeps_safe_short_words():
    assert _should_keep_latin_token("a") is True
    assert _should_keep_latin_token("I") is True


def test_low_confidence_overrides_a_coincidental_dictionary_match():
    """A short token that happens to be a real (if obscure) dictionary
    word, e.g. 'lis', should still be treated as noise if Tesseract itself
    reported low confidence for that exact occurrence — this is the gap
    plain whitelist+dictionary filtering can't close on its own."""
    low_conf = settings.OCR_ARTIFACT_MIN_CONFIDENCE - 10
    high_conf = min(99.0, settings.OCR_ARTIFACT_MIN_CONFIDENCE + 40)
    assert _should_keep_latin_token("lis", confidence=low_conf) is False
    assert _should_keep_latin_token("lis", confidence=high_conf) is True
    assert _should_keep_latin_token("lis") is True  # no confidence data -> unchanged fallback


def test_whitelisted_and_camelcase_terms_ignore_low_confidence():
    """Confidence is only a tiebreaker for the dictionary-fallback case —
    whitelisted technical terms and CamelCase/digit-mixed identifiers are
    kept regardless of how unconfident Tesseract was."""
    assert _should_keep_latin_token("RAG", confidence=1.0) is True
    assert _should_keep_latin_token("FastAPI", confidence=1.0) is True
    assert _should_keep_latin_token("GPT4", confidence=1.0) is True


def test_low_confidence_does_not_affect_longer_dictionary_words():
    """The confidence override only applies to short (<=6 char) tokens,
    where a coincidental artifact/real-word collision is plausible —
    longer real words are kept regardless."""
    assert _should_keep_latin_token("Healthcare", confidence=1.0) is True


def test_clean_ocr_artifacts_uses_confidence_map_when_given():
    text = "هذا النص يحتوي على lis كلمة غامضة"
    # Without confidence info, "lis" is a real (if obscure) dictionary word
    # and survives.
    assert "lis" in _clean_ocr_artifacts(text)
    # With low reported confidence for that exact token, it's removed.
    cleaned = _clean_ocr_artifacts(text, confidences={"lis": 5.0})
    assert "lis" not in cleaned


def test_paragraph_structure_preserved():
    text = "سطر أول به IST غير موجود\n\nسطر ثاني عادي"
    cleaned = _clean_ocr_artifacts(text)
    assert "\n\n" in cleaned  # blank-line paragraph separator untouched
