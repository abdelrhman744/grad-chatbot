"""
report_service.py

Generates a comprehensive, professional PDF report for a single previously
uploaded document.

Pipeline
--------
1. Re-download the original file from MinIO and re-extract its full text,
   page by page (services.rag_service.get_document_pages) — the whole
   document, not just chunks a retrieval query happened to surface.
2. Split the page-tagged text into map-reduce slices (bounded by
   REPORT_MAP_CHUNK_CHARS so very large documents are processed section
   by section instead of hitting token limits), keeping track of which
   page range each slice came from.
3. MAP step: for each slice, ask the LLM to extract a structured JSON
   record — a section title/summary plus key points, definitions,
   technical terms, numbers, equations, best practices, and figure/table
   mentions, each expected to come only from that slice's text.
4. Aggregate all slice records into document-wide buckets (deduplicated,
   each item tagged with the page(s) it came from).
5. REDUCE step: a handful of extra LLM calls turn the aggregated buckets
   into free-text sections — Executive Summary, Introduction,
   Relationships Between Concepts, Conclusion — using only the extracted
   facts (never the raw text again), which keeps the calls cheap even
   for huge documents and keeps hallucination risk low (the model is
   summarizing structured facts it already extracted, not inventing new
   ones).
6. Render everything as a PDF with reportlab: cover page, auto-numbered
   table of contents (multiBuild + TableOfContents), and one section per
   bucket. Arabic text is reshaped (arabic_reshaper) and bidi-reordered
   (python-bidi) and rendered with a bundled Arabic font (Amiri), since
   the built-in PDF base fonts have no Arabic glyphs.
7. Upload the finished PDF to MinIO (settings.MINIO_BUCKET_REPORTS).
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
)
from reportlab.platypus.tableofcontents import TableOfContents

from config import settings
from services import storage_service
from services.llm_provider import get_llm
from services.rag_service import detect_language, get_document_pages, retrieve

log = logging.getLogger("report_service")

llm = get_llm()

MAX_DIGEST_CHARS = 14000  # cap on how much aggregated-fact text feeds each reduce call
MAP_EXTRACT_CONCURRENCY = 5  # bounded worker pool for the per-slice MAP step
# A report needs broad topical coverage, not just the single best chat-style
# match — pull more chunks than a normal chat retrieve() call would.
TOPIC_REPORT_TOP_K = 24
# Of those candidates, drop any scoring below this fraction of the best
# match's score — otherwise a broad top_k pulls in unrelated documents just
# to fill the quota instead of reflecting only genuinely relevant material.
TOPIC_REPORT_RELEVANCE_RATIO = 0.5


# ── Arabic-capable fonts ─────────────────────────────────────────────────────

_FONTS_REGISTERED = False


def _register_fonts() -> None:
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return
    font_dir = Path(settings.REPORT_FONT_DIR)
    regular, bold = font_dir / "Amiri-Regular.ttf", font_dir / "Amiri-Bold.ttf"
    if regular.is_file():
        pdfmetrics.registerFont(TTFont("Amiri", str(regular)))
    if bold.is_file():
        pdfmetrics.registerFont(TTFont("Amiri-Bold", str(bold)))
    _FONTS_REGISTERED = True


def _shape(text: str) -> str:
    """Reshape + bidi-reorder Arabic text for correct visual rendering."""
    if not text:
        return text
    return get_display(arabic_reshaper.reshape(text))


def _shape_mixed(text: str) -> str:
    return "\n".join(_shape(line) if line.strip() else line for line in text.splitlines())


def _escape(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ── Page-aware chunking ──────────────────────────────────────────────────────

def _chunk_pages(pages: List[dict], size: int) -> List[dict]:
    """Group consecutive pages into slices up to ~`size` chars, keeping the
    page range each slice spans so facts can be traced back to a page."""
    chunks, current_text, p_start, p_end = [], "", None, None

    for p in pages:
        text = p["text"]
        if p_start is None:
            p_start = p["page"]

        if len(current_text) + len(text) + 2 > size and current_text:
            chunks.append({"text": current_text, "page_start": p_start, "page_end": p_end})
            current_text, p_start = text, p["page"]
        else:
            current_text = f"{current_text}\n\n{text}" if current_text else text

        p_end = p["page"]

    if current_text:
        chunks.append({"text": current_text, "page_start": p_start, "page_end": p_end})

    return chunks


def _page_ref(page_start: int, page_end: int) -> str:
    return f"p. {page_start}" if page_start == page_end else f"pp. {page_start}-{page_end}"


# ── MAP step: structured per-slice extraction ────────────────────────────────

_MAP_SCHEMA_HINT = (
    '{"section_title": "...", "summary": "...", "key_points": ["..."], '
    '"definitions": [{"term": "...", "definition": "..."}], "technical_terms": ["..."], '
    '"numbers": ["..."], "equations": ["..."], "best_practices": ["..."], '
    '"figures_tables": ["..."]}'
)


def _map_extract(slice_text: str, lang: str, index: int, total: int) -> dict:
    if lang == "ar":
        prompt = f"""أنت تحلل الجزء {index} من {total} من مستند تقني. استخرج المعلومات التالية بدقة، بالاعتماد فقط على النص المعطى، بدون أي إضافة أو تخمين:

النص:
{slice_text}

أرجع كائن JSON واحد فقط بالضبط بهذا الشكل (بدون أي نص قبله أو بعده، وبدون Markdown):
{_MAP_SCHEMA_HINT}

حيث:
- section_title: عنوان قصير (3-8 كلمات) يصف موضوع هذا الجزء.
- summary: فقرة من 2-4 جمل تلخص هذا الجزء فقط.
- key_points: أهم النقاط الواردة في هذا الجزء (قائمة نصوص قصيرة).
- definitions: أي تعريفات لمصطلحات وردت حرفيًا في النص (term + definition).
- technical_terms: المصطلحات التقنية المهمة كما وردت بالضبط.
- numbers: أي أرقام أو إحصائيات أو تواريخ مهمة، مع سياقها.
- equations: أي معادلات أو صيغ رياضية وردت حرفيًا (فارغة إذا لا يوجد).
- best_practices: أي نصائح أو ممارسات موصى بها ذُكرت صراحة (فارغة إذا لا يوجد).
- figures_tables: أي إشارات لأشكال أو جداول مذكورة في النص (مثل "الشكل 2 يوضح..."), فارغة إذا لا يوجد.
كل الحقول مطلوبة؛ استخدم قائمة فارغة [] إذا لم تجد شيئًا مناسبًا. لا تخترع أي معلومة غير موجودة في النص."""
    else:
        prompt = f"""You are analyzing part {index} of {total} of a technical document. Extract the following information precisely, using ONLY the given text — do not add or guess anything.

Text:
{slice_text}

Return exactly one JSON object in this shape (no text before/after, no Markdown):
{_MAP_SCHEMA_HINT}

Where:
- section_title: a short title (3-8 words) describing this part's topic.
- summary: a 2-4 sentence summary of this part only.
- key_points: the most important points in this part (short text list).
- definitions: any term definitions given verbatim in the text (term + definition).
- technical_terms: important technical terms exactly as written.
- numbers: any important figures, statistics, or dates, with brief context.
- equations: any equations/formulas given verbatim (empty if none).
- best_practices: any explicitly stated recommendations/best practices (empty if none).
- figures_tables: any references to figures/tables mentioned in the text (e.g. "Figure 2 shows..."), empty if none.
All fields are required; use an empty list [] if nothing applies. Never invent information not present in the text."""

    try:
        raw = str(llm.invoke(prompt)).strip()
        data = _parse_json_object(raw)
    except Exception as e:
        log.error(f"[report] map step {index}/{total} failed: {e}")
        data = {}

    return {
        "section_title": data.get("section_title") or (f"Section {index}" if lang != "ar" else f"القسم {index}"),
        "summary": data.get("summary", ""),
        "key_points": _as_str_list(data.get("key_points")),
        "definitions": data.get("definitions") if isinstance(data.get("definitions"), list) else [],
        "technical_terms": _as_str_list(data.get("technical_terms")),
        "numbers": _as_str_list(data.get("numbers")),
        "equations": _as_str_list(data.get("equations")),
        "best_practices": _as_str_list(data.get("best_practices")),
        "figures_tables": _as_str_list(data.get("figures_tables")),
    }


def _as_str_list(value) -> list:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def _parse_json_object(raw: str) -> dict:
    """Best-effort JSON parsing: strips Markdown code fences and pulls out
    the first {...} block if the model added stray text around it."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


# ── Aggregation ──────────────────────────────────────────────────────────────

def _dedupe_with_pages(items_by_page: List[tuple]) -> List[str]:
    """items_by_page: [(text, page_ref_str), ...] -> deduplicated
    '<text> (<page_ref>)' strings, keeping first-seen order."""
    seen, out = set(), []
    for text, page_ref in items_by_page:
        key = re.sub(r"\s+", " ", text.strip().lower())[:200]
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(f"{text.strip()} ({page_ref})" if page_ref else text.strip())
    return out


def _aggregate(map_results: List[dict]) -> dict:
    key_points, technical_terms, numbers, equations = [], [], [], []
    best_practices, figures_tables = [], []
    definitions_seen: dict = {}

    for r in map_results:
        ref = r["page_ref"]
        key_points += [(kp, ref) for kp in r["key_points"]]
        technical_terms += [(t, ref) for t in r["technical_terms"]]
        numbers += [(n, ref) for n in r["numbers"]]
        equations += [(e, ref) for e in r["equations"]]
        best_practices += [(b, ref) for b in r["best_practices"]]
        figures_tables += [(f, ref) for f in r["figures_tables"]]

        for d in r["definitions"]:
            if not isinstance(d, dict):
                continue
            term = str(d.get("term", "")).strip()
            definition = str(d.get("definition", "")).strip()
            if term and term.lower() not in definitions_seen:
                definitions_seen[term.lower()] = {"term": term, "definition": definition, "page": ref}

    return {
        "key_points": _dedupe_with_pages(key_points),
        "technical_terms": _dedupe_with_pages(technical_terms),
        "numbers": _dedupe_with_pages(numbers),
        "equations": _dedupe_with_pages(equations),
        "best_practices": _dedupe_with_pages(best_practices),
        "figures_tables": _dedupe_with_pages(figures_tables),
        "definitions": list(definitions_seen.values()),
    }


# ── REDUCE step: whole-document narrative sections ───────────────────────────

def _digest(map_results: List[dict]) -> str:
    parts = [f"[{r['page_ref']}] {r['section_title']}: {r['summary']}" for r in map_results if r["summary"]]
    text = "\n".join(parts)
    if len(text) > MAX_DIGEST_CHARS:
        text = text[:MAX_DIGEST_CHARS] + "\n...(truncated — document continues)"
    return text


def _reduce_narrative(digest: str, aggregated: dict, filename: str, lang: str, kind: str) -> str:
    facts = "\n".join(f"- {kp}" for kp in aggregated["key_points"][:60])

    prompts = {
        "executive_summary": {
            "ar": f"""بالاعتماد فقط على ملخصات أجزاء المستند "{filename}" أدناه، اكتب ملخصًا تنفيذيًا احترافيًا من 5-7 جمل يغطي الغرض من المستند وأهم ما جاء فيه. لا تخترع أي معلومة غير مذكورة.

ملخصات الأجزاء:
{digest}""",
            "en": f"""Based only on the section summaries of document "{filename}" below, write a professional executive summary (5-7 sentences) covering the document's purpose and most important content. Do not invent information not mentioned.

Section summaries:
{digest}""",
        },
        "introduction": {
            "ar": f"""بالاعتماد فقط على ملخصات أجزاء المستند "{filename}" أدناه، اكتب مقدمة من 3-5 جمل تشرح موضوع المستند ونطاقه والجمهور المستهدف منه إن أمكن استنتاجه.

ملخصات الأجزاء:
{digest}""",
            "en": f"""Based only on the section summaries of document "{filename}" below, write a 3-5 sentence introduction explaining the document's subject, scope, and intended audience if inferable.

Section summaries:
{digest}""",
        },
        "relationships": {
            "ar": f"""بالاعتماد فقط على النقاط الرئيسية التالية المستخرجة من المستند "{filename}"، اشرح في فقرة أو فقرتين العلاقات والروابط بين أهم المفاهيم الواردة (كيف يرتبط بعضها ببعض، أو يعتمد بعضها على بعض). إذا لم تكن هناك علاقات واضحة بين المفاهيم، اذكر ذلك بإيجاز بدلاً من اختلاق روابط.

النقاط الرئيسية:
{facts}""",
            "en": f"""Based only on the following key points extracted from document "{filename}", explain in one or two paragraphs how the main concepts relate to or depend on each other. If no clear relationships exist between the concepts, briefly say so rather than inventing connections.

Key points:
{facts}""",
        },
        "conclusion": {
            "ar": f"""بالاعتماد فقط على ملخصات الأجزاء التالية من المستند "{filename}"، اكتب خاتمة احترافية من 3-5 جمل تلخص أهم النتائج أو الدروس المستفادة من المستند.

ملخصات الأجزاء:
{digest}""",
            "en": f"""Based only on the following section summaries of document "{filename}", write a professional conclusion (3-5 sentences) summarizing the document's key takeaways.

Section summaries:
{digest}""",
        },
    }

    prompt = prompts[kind]["ar" if lang == "ar" else "en"]

    try:
        return str(llm.invoke(prompt)).strip()
    except Exception as e:
        log.error(f"[report] reduce step '{kind}' failed: {e}")
        return ""


# ── Full pipeline ────────────────────────────────────────────────────────────

def build_report_data(filename: str) -> dict:
    pages = get_document_pages(filename)
    if not pages:
        raise ValueError(f"No extractable text found in '{filename}'.")

    full_text_sample = " ".join(p["text"] for p in pages)[:2000]
    lang = detect_language(full_text_sample)

    raw_chunks = _chunk_pages(pages, settings.REPORT_MAP_CHUNK_CHARS)
    total = len(raw_chunks)

    def _map_one(indexed_chunk) -> dict:
        i, chunk = indexed_chunk
        extracted = _map_extract(chunk["text"], lang, i + 1, total)
        extracted["page_ref"] = _page_ref(chunk["page_start"], chunk["page_end"])
        return extracted

    # Each slice's MAP-extract call only depends on that slice's own text —
    # run them on a bounded worker pool instead of one Groq round-trip at a
    # time. For a large document this was previously the single biggest
    # contributor to report latency (~N sequential calls, N = slice count);
    # it's now ~N / MAP_EXTRACT_CONCURRENCY. pool.map preserves input order,
    # so section ordering in the final report is unaffected.
    if total:
        with ThreadPoolExecutor(max_workers=min(MAP_EXTRACT_CONCURRENCY, total)) as pool:
            map_results = list(pool.map(_map_one, enumerate(raw_chunks)))
    else:
        map_results = []

    aggregated = _aggregate(map_results)
    digest = _digest(map_results)

    # The 4 REDUCE narrative sections share the same digest/aggregated
    # input and only differ by which section they're writing — independent
    # calls, so run them concurrently too instead of 4 sequential ones.
    narrative_kinds = ["executive_summary", "introduction", "relationships", "conclusion"]
    with ThreadPoolExecutor(max_workers=len(narrative_kinds)) as pool:
        narratives = dict(zip(
            narrative_kinds,
            pool.map(
                lambda kind: _reduce_narrative(digest, aggregated, filename, lang, kind),
                narrative_kinds,
            ),
        ))

    return {
        "filename": filename,
        "language": lang,
        "page_count": pages[-1]["page"] if pages else 0,
        "section_count": total,
        "sections": [
            {
                "title": r["section_title"],
                "page_ref": r["page_ref"],
                "summary": r["summary"],
                "key_points": r["key_points"],
            }
            for r in map_results
        ],
        "executive_summary": narratives["executive_summary"],
        "introduction": narratives["introduction"],
        "relationships": narratives["relationships"],
        "conclusion": narratives["conclusion"],
        **aggregated,
    }


def _topic_is_covered(topic: str, chunks: List[dict], lang: str) -> bool:
    """
    LLM relevance gate for topic-scoped reports: does the retrieved
    material actually contain information about `topic`, or is it just the
    least-irrelevant match retrieval could find? See the call site in
    build_topic_report_data() for why a lexical/embedding score threshold
    alone isn't trusted for this decision.
    """
    sample = "\n\n".join(c["text"][:500] for c in chunks[:6])

    if lang == "ar":
        prompt = f"""فيما يلي مقتطفات مسترجعة من مستندات مرفوعة، والموضوع المطلوب إنشاء تقرير عنه.
هل تحتوي هذه المقتطفات فعليًا على معلومات ذات صلة مباشرة بالموضوع التالي؟ أجب بكلمة واحدة فقط: نعم أو لا.

الموضوع: {topic}

المقتطفات:
{sample}"""
    else:
        prompt = f"""Below are excerpts retrieved from uploaded documents, and a topic requested for a report.
Do these excerpts actually contain information directly relevant to the topic below? Answer with exactly one word: yes or no.

Topic: {topic}

Excerpts:
{sample}"""

    try:
        out = str(llm.invoke(prompt)).strip().lower()
        return out.startswith(("yes", "نعم"))
    except Exception as e:
        log.warning(f"[report] topic relevance check failed, assuming relevant: {e}")
        return True  # fail open — a transient LLM error shouldn't block a report


def build_topic_report_data(topic: str, document: Optional[str] = None) -> dict:
    """
    Topic-scoped counterpart to build_report_data(): instead of re-reading
    one whole document end to end, retrieve the chunks most relevant to
    `topic` — across the entire uploaded knowledge base, or within
    `document` only if one was named — and run the same MAP/REDUCE pipeline
    over the retrieved chunk text instead of over whole-document pages.

    Raises ValueError if too little relevant material is found, so the
    caller can tell the user that plainly instead of generating a report
    from unrelated or insufficient content.
    """
    topic = (topic or "").strip()
    if not topic:
        raise ValueError("No topic was given to build a report from.")

    lang = detect_language(topic)

    chunks = retrieve(topic, lang=lang, top_k=TOPIC_REPORT_TOP_K, source_filter=document)

    # retrieve() with a large top_k returns its best top_k candidates
    # regardless of whether that many are genuinely relevant — for a chat
    # answer (top_k~5) that's rarely visible, but at TOPIC_REPORT_TOP_K it
    # would otherwise pad the report with barely-related chunks from
    # unrelated documents just to fill the quota. Keep only chunks scoring
    # reasonably close to the best match found for this topic.
    if chunks:
        top_score = max(c["metadata"].get("relevance_score", 0.0) for c in chunks)
        min_score = max(top_score * TOPIC_REPORT_RELEVANCE_RATIO, settings.CONFIDENCE_THRESHOLD)
        chunks = [c for c in chunks if c["metadata"].get("relevance_score", 0.0) >= min_score]

    insufficient_msg = (
        f"لم يتم العثور على معلومات كافية عن '{topic}' في المستندات المرفوعة."
        if lang == "ar" else
        f"Not enough information about '{topic}' was found in the uploaded documents."
    )

    if not chunks:
        raise ValueError(insufficient_msg)

    # The relevance-score filter above only trims obviously-weak padding; it
    # cannot reliably tell "on-topic" from "the least-bad match among
    # entirely unrelated documents" (the same limitation noted on
    # CONFIDENCE_THRESHOLD in rag_service.py — lexical/embedding scores
    # alone aren't a reliable topic classifier). Retrieval always returns
    # *something* if the knowledge base isn't empty, even for a topic
    # nothing covers. An explicit LLM relevance judgment is the real gate
    # here, since — unlike a chat answer, where build_prompt()'s grounding
    # rule lets the model decline per-question — nothing downstream in the
    # report pipeline would otherwise refuse to write up unrelated content.
    if not _topic_is_covered(topic, chunks, lang):
        raise ValueError(insufficient_msg)

    sources = sorted({c["metadata"].get("source", "?") for c in chunks})

    # Reuse the same page-aware slicing build_report_data() uses, but keyed
    # by retrieval rank instead of physical page order — retrieved chunks
    # can come from different documents/pages entirely, so each chunk is
    # its own "page" here, prefixed with its source filename so the MAP
    # step (and a human skimming the PDF) can still tell where it came from.
    pseudo_pages = [
        {"page": i + 1, "text": f"[{c['metadata'].get('source', '?')}] {c['text']}"}
        for i, c in enumerate(chunks)
    ]
    raw_chunks = _chunk_pages(pseudo_pages, settings.REPORT_MAP_CHUNK_CHARS)
    total = len(raw_chunks)

    def _map_one(indexed_chunk) -> dict:
        i, chunk = indexed_chunk
        extracted = _map_extract(chunk["text"], lang, i + 1, total)
        extracted["page_ref"] = _page_ref(chunk["page_start"], chunk["page_end"])
        return extracted

    if total:
        with ThreadPoolExecutor(max_workers=min(MAP_EXTRACT_CONCURRENCY, total)) as pool:
            map_results = list(pool.map(_map_one, enumerate(raw_chunks)))
    else:
        map_results = []

    aggregated = _aggregate(map_results)
    digest = _digest(map_results)

    narrative_kinds = ["executive_summary", "introduction", "relationships", "conclusion"]
    with ThreadPoolExecutor(max_workers=len(narrative_kinds)) as pool:
        narratives = dict(zip(
            narrative_kinds,
            pool.map(
                lambda kind: _reduce_narrative(digest, aggregated, topic, lang, kind),
                narrative_kinds,
            ),
        ))

    return {
        "filename": topic,
        "sources": sources,
        "language": lang,
        "page_count": total,
        "section_count": total,
        "sections": [
            {
                "title": r["section_title"],
                "page_ref": r["page_ref"],
                "summary": r["summary"],
                "key_points": r["key_points"],
            }
            for r in map_results
        ],
        "executive_summary": narratives["executive_summary"],
        "introduction": narratives["introduction"],
        "relationships": narratives["relationships"],
        "conclusion": narratives["conclusion"],
        **aggregated,
    }


# ── PDF rendering ────────────────────────────────────────────────────────────

class _ReportDocTemplate(BaseDocTemplate):
    """BaseDocTemplate that records (heading text, page number) for every
    flowable styled as 'SectionHeading', so the TableOfContents flowable
    can resolve real page numbers via a second build pass (multiBuild)."""

    def afterFlowable(self, flowable):
        if getattr(flowable, "style", None) and flowable.style.name == "SectionHeading":
            self.notify("TOCEntry", (0, flowable.getPlainText(), self.page))


def _styles(lang: str) -> dict:
    _register_fonts()
    base = getSampleStyleSheet()
    is_ar = lang == "ar"
    font, font_bold = ("Amiri", "Amiri-Bold") if is_ar else ("Helvetica", "Helvetica-Bold")
    align = TA_RIGHT if is_ar else TA_LEFT

    return {
        "cover_title": ParagraphStyle(
            "CoverTitle", parent=base["Title"], fontName=font_bold, fontSize=26,
            alignment=TA_CENTER, textColor=colors.HexColor("#0f172a"), spaceAfter=10, leading=32,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle", parent=base["Normal"], fontName=font, fontSize=14,
            alignment=TA_CENTER, textColor=colors.HexColor("#0f766e"), spaceAfter=6,
        ),
        "cover_meta": ParagraphStyle(
            "CoverMeta", parent=base["Normal"], fontName=font, fontSize=10.5,
            alignment=TA_CENTER, textColor=colors.HexColor("#555555"), spaceAfter=3,
        ),
        "toc_title": ParagraphStyle(
            "TOCTitle", parent=base["Title"], fontName=font_bold, fontSize=20,
            alignment=align, spaceAfter=14,
        ),
        "toc_entry": ParagraphStyle(
            "TOCLevel0", fontName=font, fontSize=11, leading=18, alignment=align,
        ),
        "heading": ParagraphStyle(
            "SectionHeading", parent=base["Heading2"], fontName=font_bold, fontSize=15,
            alignment=align, textColor=colors.HexColor("#0f766e"), spaceBefore=14, spaceAfter=8,
        ),
        "subheading": ParagraphStyle(
            "SubHeading", parent=base["Heading3"], fontName=font_bold, fontSize=12,
            alignment=align, textColor=colors.HexColor("#1a1a1a"), spaceBefore=10, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["Normal"], fontName=font, fontSize=10.5, leading=16, alignment=align,
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=base["Normal"], fontName=font, fontSize=10.5, leading=15,
            alignment=align, leftIndent=0 if is_ar else 14, rightIndent=14 if is_ar else 0, spaceAfter=3,
        ),
        "mono": ParagraphStyle(
            "Mono", parent=base["Normal"], fontName="Helvetica", fontSize=9.5, leading=14,
            alignment=TA_LEFT, leftIndent=14, textColor=colors.HexColor("#333333"), spaceAfter=3,
        ),
    }


def _p(text: str, style: ParagraphStyle, lang: str) -> Paragraph:
    shaped = _shape_mixed(text) if lang == "ar" else text
    return Paragraph(_escape(shaped).replace("\n", "<br/>"), style)


def _bullets(items: List[str], style: ParagraphStyle, lang: str) -> list:
    bullet = " •" if lang == "ar" else "• "
    out = []
    for item in items:
        content = f"{item}{bullet}" if lang == "ar" else f"{bullet}{item}"
        out.append(_p(content, style, lang))
    return out


_LABELS = {
    "ar": {
        "cover_subtitle": "تقرير تقني احترافي",
        "generated": "تاريخ الإنشاء",
        "pages": "عدد الصفحات",
        "sections": "عدد الأقسام",
        "toc": "جدول المحتويات",
        "executive_summary": "الملخص التنفيذي",
        "introduction": "المقدمة",
        "detailed_sections": "شرح تفصيلي للمحتوى",
        "key_concepts": "المفاهيم الرئيسية",
        "definitions": "التعريفات",
        "technical_terms": "المصطلحات التقنية المهمة",
        "numbers": "الأرقام والإحصائيات المهمة",
        "equations": "المعادلات",
        "figures_tables": "الأشكال والجداول المذكورة",
        "best_practices": "أفضل الممارسات",
        "relationships": "العلاقات بين المفاهيم",
        "conclusion": "الخاتمة",
        "references": "المراجع",
        "ref_note": "تم إنشاء هذا التقرير تلقائيًا من المستند المرفوع. أرقام الصفحات المذكورة تشير إلى صفحات المستند الأصلي.",
        "source_doc": "المستند المصدر",
    },
    "en": {
        "cover_subtitle": "Professional Technical Report",
        "generated": "Generated",
        "pages": "Pages",
        "sections": "Sections",
        "toc": "Table of Contents",
        "executive_summary": "Executive Summary",
        "introduction": "Introduction",
        "detailed_sections": "Detailed Content Overview",
        "key_concepts": "Key Concepts",
        "definitions": "Definitions",
        "technical_terms": "Important Technical Terms",
        "numbers": "Important Numbers & Statistics",
        "equations": "Equations",
        "figures_tables": "Figures & Tables Referenced",
        "best_practices": "Best Practices",
        "relationships": "Relationships Between Concepts",
        "conclusion": "Conclusion",
        "references": "References",
        "ref_note": "This report was generated automatically from the uploaded document. Page numbers refer to the original document's pages.",
        "source_doc": "Source document",
    },
}


def render_report_pdf(data: dict) -> bytes:
    lang = data.get("language", "en")
    is_ar = lang == "ar"
    styles = _styles(lang)
    labels = _LABELS["ar" if is_ar else "en"]
    filename = data["filename"]

    buf = io.BytesIO()
    doc = _ReportDocTemplate(
        buf, pagesize=A4,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame])])

    toc = TableOfContents()
    toc.levelStyles = [styles["toc_entry"]]

    story = []

    # ── Cover page ───────────────────────────────────────────────────────
    story.append(Spacer(1, 6 * cm))
    story.append(_p(os.path.splitext(filename)[0], styles["cover_title"], lang))
    story.append(_p(labels["cover_subtitle"], styles["cover_subtitle"], lang))
    story.append(Spacer(1, 1 * cm))
    story.append(HRFlowable(width="60%", color=colors.HexColor("#0f766e"), thickness=1.2, hAlign="CENTER"))
    story.append(Spacer(1, 0.6 * cm))
    # Topic-scoped reports (build_topic_report_data) set "sources" to the
    # list of contributing filenames; whole-document reports don't set it,
    # so this falls back to the single filename exactly as before.
    contributing_sources = data.get("sources") or [filename]
    source_line = ", ".join(contributing_sources) if len(contributing_sources) > 1 else filename
    story.append(_p(f"{labels['source_doc']}: {source_line}", styles["cover_meta"], lang))
    story.append(_p(f"{labels['generated']}: {time.strftime('%Y-%m-%d %H:%M UTC')}", styles["cover_meta"], lang))
    story.append(_p(f"{labels['pages']}: {data.get('page_count', '?')}  |  {labels['sections']}: {data.get('section_count', '?')}", styles["cover_meta"], lang))
    story.append(PageBreak())

    # ── Table of contents ────────────────────────────────────────────────
    story.append(_p(labels["toc"], styles["toc_title"], lang))
    story.append(toc)
    story.append(PageBreak())

    def heading(text: str):
        story.append(_p(text, styles["heading"], lang))

    def body(text: str):
        if text.strip():
            story.append(_p(text, styles["body"], lang))

    def bullets(items: List[str]):
        story.extend(_bullets(items, styles["bullet"], lang))

    # ── Executive summary & introduction ─────────────────────────────────
    heading(labels["executive_summary"])
    body(data.get("executive_summary", ""))

    heading(labels["introduction"])
    body(data.get("introduction", ""))

    # ── Detailed section-by-section overview ─────────────────────────────
    heading(labels["detailed_sections"])
    for section in data.get("sections", []):
        story.append(_p(f"{section['title']}  ({section['page_ref']})", styles["subheading"], lang))
        body(section["summary"])
        if section["key_points"]:
            bullets(section["key_points"])

    # ── Key concepts / definitions / terms / numbers / equations ─────────
    if data.get("key_points"):
        heading(labels["key_concepts"])
        bullets(data["key_points"][:40])

    if data.get("definitions"):
        heading(labels["definitions"])
        for d in data["definitions"][:40]:
            label = f"{d['term']} ({d['page']})" if d.get("page") else d["term"]
            story.append(_p(label, styles["subheading"], lang))
            body(d.get("definition", ""))

    if data.get("technical_terms"):
        heading(labels["technical_terms"])
        bullets(data["technical_terms"][:50])

    if data.get("numbers"):
        heading(labels["numbers"])
        bullets(data["numbers"][:40])

    if data.get("equations"):
        heading(labels["equations"])
        for eq in data["equations"][:30]:
            story.append(Paragraph(_escape(eq), styles["mono"]))

    if data.get("figures_tables"):
        heading(labels["figures_tables"])
        bullets(data["figures_tables"][:30])

    if data.get("best_practices"):
        heading(labels["best_practices"])
        bullets(data["best_practices"][:30])

    # ── Relationships, conclusion, references ────────────────────────────
    if data.get("relationships", "").strip():
        heading(labels["relationships"])
        body(data["relationships"])

    heading(labels["conclusion"])
    body(data.get("conclusion", ""))

    heading(labels["references"])
    body(f"{labels['source_doc']}: {source_line}")
    body(labels["ref_note"])

    doc.multiBuild(story)
    return buf.getvalue()


# ── Public entrypoint ────────────────────────────────────────────────────────

def _report_object_name(filename: str) -> str:
    stem = re.sub(r'[<>:"/\\|?*]', "_", os.path.splitext(filename)[0]).strip() or "document"
    return f"{stem}_report.pdf"


def _topic_report_object_name(topic: str) -> str:
    slug = re.sub(r"\s+", "_", topic.strip())
    slug = re.sub(r'[<>:"/\\|?*]', "_", slug)[:80].strip("_") or "topic"
    return f"{slug}_report.pdf"


def generate_report(filename: str) -> dict:
    """
    Generate (or regenerate) the comprehensive PDF report for `filename`,
    store it in MinIO, and return {"object_name", "download_url", "language"}.
    """
    data = build_report_data(filename)
    pdf_bytes = render_report_pdf(data)

    object_name = _report_object_name(filename)
    storage_service.upload_bytes(
        settings.MINIO_BUCKET_REPORTS, object_name, pdf_bytes, content_type="application/pdf"
    )

    return {
        "object_name": object_name,
        "download_url": storage_service.presigned_url(settings.MINIO_BUCKET_REPORTS, object_name),
        "language": data["language"],
    }


def generate_topic_report(topic: str, document: Optional[str] = None) -> dict:
    """
    Generate a topic-scoped PDF report: retrieves the material most
    relevant to `topic` (across all uploaded documents, or within
    `document` only if given) and reuses the same MAP/REDUCE/PDF pipeline
    as generate_report(). Raises ValueError if too little relevant
    material was found for the topic — the caller (ReportTool) surfaces
    that as a plain "not enough information" answer instead of a report.
    """
    data = build_topic_report_data(topic, document=document)
    pdf_bytes = render_report_pdf(data)

    object_name = _topic_report_object_name(topic)
    storage_service.upload_bytes(
        settings.MINIO_BUCKET_REPORTS, object_name, pdf_bytes, content_type="application/pdf"
    )

    return {
        "object_name": object_name,
        "download_url": storage_service.presigned_url(settings.MINIO_BUCKET_REPORTS, object_name),
        "language": data["language"],
    }


def get_report_bytes(object_name: str) -> bytes:
    return storage_service.download_bytes(settings.MINIO_BUCKET_REPORTS, object_name)
