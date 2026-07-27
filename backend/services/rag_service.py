"""
rag_service.py

Core Retrieval-Augmented-Generation engine.

Responsibilities:
- Document ingestion (loading, OCR fallback, chunking, embedding, storage)
- Query-variant generation (Arabic/English normalization, translation, typo
  correction) for robust bilingual retrieval
- Lexical reranking on top of vector search
- Direct question answering (`ask_question`, used by the simple /chat route)
- A small *public* retrieval/generation API (`retrieve`, `generate_answer`,
  `summarize`, `compare`) that the agent's tools call into, so the agent
  reuses this exact pipeline instead of re-implementing it.

All heavy text-processing logic (query variants, reranking, confidence,
prompts) is preserved from the original prototype; only configuration,
the quiz feature, and the internal/external API surface changed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import time
from collections import Counter
from functools import lru_cache
from typing import Any, List, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from services.db_service import get_client, get_collection_name, ensure_collection
from services.embeddings_provider import get_embeddings
from services.llm_provider import GroqLLM, get_llm as _get_shared_llm
from services.ocr_service import perform_ocr_pdf_bytes, perform_ocr_image_bytes
from services import storage_service

log = logging.getLogger("rag_service")

# ── Models ──────────────────────────────────────────────────────────────────
# `embeddings` powers Qdrant indexing/retrieval; `llm` powers every text
# generation call in this module (answer generation, translation,
# rephrasing, spelling correction, summarization, comparison). Both are
# shared singletons, matching the previous Ollama-based setup so the rest
# of this file — and the agent/memory modules that call `llm.invoke(...)`
# — required no further changes.
embeddings = get_embeddings()

llm = _get_shared_llm()


def get_llm() -> GroqLLM:
    """Expose the shared LLM instance (used by the agent and memory summarizer)."""
    return llm


# ── Vector DB state ───────────────────────────────────────────────────────────
_vector_db: QdrantVectorStore | None = None
_retriever = None


def _get_vector_db() -> QdrantVectorStore:
    global _vector_db
    if _vector_db is None:
        ensure_collection(embeddings)
        _vector_db = QdrantVectorStore(
            client=get_client(),
            collection_name=get_collection_name(),
            embedding=embeddings,
        )
    return _vector_db


def _refresh_retriever():
    global _retriever, _vector_db
    if _vector_db is None:
        _retriever = None
        return
    _retriever = _vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.RETRIEVER_K},
    )


def load_existing_db() -> bool:
    """Call once at startup to attach to an existing Qdrant collection."""
    global _vector_db
    try:
        client = get_client()
        client.get_collection(get_collection_name())
        _vector_db = QdrantVectorStore(
            client=client,
            collection_name=get_collection_name(),
            embedding=embeddings,
        )
        count = client.get_collection(get_collection_name()).points_count
        log.info(f"Loaded existing DB — {count} vectors")
        _refresh_retriever()
        return True
    except Exception as e:
        log.warning(f"Could not load existing DB: {e}")
        _vector_db = None
        return False


def is_ready() -> bool:
    """Whether the vector store has been initialised and is queryable."""
    if _retriever is None:
        load_existing_db()
    return _retriever is not None


# ── Text utilities ─────────────────────────────────────────────────────────────

AR_STOPWORDS = {
    "ما","ماذا","كيف","هل","في","من","على","الى","إلى","عن","و","او","أو",
    "هو","هي","هم","هذه","هذا","ذلك","تلك","كل","بين","مع","ل","ال","اي","أي",
    "الذي","التي","ثم","بعد","قبل","هناك","هنا","انه","إنه","ان","إن","كان","كانت",
    "دي","ده","دا","يعني","بس","اووي","اوي","كده","كدا",
    "عايز","عاوز","ممكن","لو","لو سمحت","حابب","ابي","ودي","عندي",
    "فيه","فية","فيها","منين","فين","امتى","ليه","ازاي",
}

EN_STOPWORDS = {
    "what","how","is","are","the","a","an","of","to","in","on","for","and",
    "or","between","about","explain","tell","me","does","do","be","types",
    "define","definition","difference","compare","give","show","list","summary",
    "this","that","these","those","with","from","into","by","it","please",
    "can","could","would","should","may","might","will","shall",
    "i","you","we","they","my","your","our","their","its",
}


def _clean(text) -> str:
    return str(text).strip() if text is not None else ""


def detect_language(text: str) -> str:
    text = text or ""

    ar_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    en_chars = len(re.findall(r"[a-zA-Z]", text))

    if ar_chars == 0 and en_chars == 0:
        return "en"

    return "ar" if ar_chars >= en_chars else "en"


def _normalize_arabic(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", text)
    text = re.sub(r"[إأآاٱ]", "ا", text)
    text = re.sub(r"[ىیي]", "ي", text)
    text = re.sub(r"[ؤو]", "و", text)
    text = re.sub(r"[ةه]", "ه", text)
    text = re.sub(r"[ئ]", "ي", text)
    text = re.sub(r"ـ+", "", text)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize(text: str) -> str:
    if not text:
        return ""
    if detect_language(text) == "ar":
        return _normalize_arabic(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _keywords(text: str, lang: str) -> List[str]:
    words = _normalize(text).split()
    stops = AR_STOPWORDS if lang == "ar" else EN_STOPWORDS
    return [w for w in words if w not in stops and len(w) > 2]


def _ngrams(text: str, n: int = 2) -> List[str]:
    words = _normalize(text).split()
    return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)] if len(words) >= n else []


def _is_meaningful(text: str) -> bool:
    c = _clean(text)
    return len(c) >= 15 and bool(re.search(r"[\u0600-\u06FFa-zA-Z]", c))


def _deduplicate(docs: List[Document]) -> List[Document]:
    seen, out = set(), []
    for d in docs:
        key = _clean(d.page_content)[:1000]
        if key and key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _deduplicate_retrieved(docs: List[Document]) -> List[Document]:
    seen, out = set(), []
    for d in docs:
        key = (_clean(d.page_content)[:700], d.metadata.get("source", ""), d.metadata.get("page", -1))
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _enrich(text: str) -> str:
    text = _clean(text)
    if not text:
        return ""
    lang = detect_language(text)
    blocks = [text]
    if lang == "ar":
        norm = _normalize_arabic(text)
        if norm and norm != text:
            blocks += ["[normalized_arabic]", norm]
    else:
        lower = text.lower()
        if lower != text:
            blocks += ["[lowercase]", lower]
    return "\n\n".join(b for b in blocks if _clean(b))


# ── Translation & query variants ─────────────────────────────────────────────

@lru_cache(maxsize=512)
def _translate(text: str, target_lang: str) -> str:
    text = _clean(text)
    if not text:
        return ""
    prompt = (
        f"Translate this Arabic text to English. Return ONLY the translation:\n{text}"
        if target_lang == "en"
        else f"Translate this English text to Arabic. Return ONLY the translation:\n{text}"
    )
    try:
        out = str(llm.invoke(prompt)).strip()
        return _clean(re.split(r"\n\n", out)[0])
    except Exception as e:
        log.warning(f"Translation failed: {e}")
        return ""


@lru_cache(maxsize=256)
def _rephrase(query: str, lang: str) -> str:
    if not query:
        return ""
    prompt = (
        f"Rephrase this question differently using synonyms (one sentence only):\n{query}"
        if lang == "en"
        else f"أعِد صياغة هذا السؤال بأسلوب مختلف (جملة واحدة فقط):\n{query}"
    )
    try:
        return _clean(str(llm.invoke(prompt)).strip().split("\n")[0])
    except Exception:
        return ""


def _is_mixed_language(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or "")) and bool(re.search(r"[a-zA-Z]", text or ""))


def _loose_arabic(text: str) -> str:
    """Extra-tolerant Arabic normalization for typo-heavy user questions."""
    text = _normalize_arabic(text)
    text = re.sub(r"\bال", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _loose_english(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=256)
def _fix_query_spelling(query: str, lang: str) -> str:
    """
    Lightweight LLM spelling correction.
    Used only to create extra retrieval variants; it never replaces the user's original question.
    """
    query = _clean(query)
    if not query:
        return ""

    if lang == "ar":
        prompt = f"""صحح الأخطاء الإملائية في السؤال التالي بدون تغيير المعنى.
أعد السؤال فقط بدون شرح أو مقدمات:

{query}"""
    else:
        prompt = f"""Correct spelling mistakes in this question without changing its meaning.
Return only the corrected question, no explanation:

{query}"""

    try:
        out = str(llm.invoke(prompt)).strip()
        return _clean(out.splitlines()[0])
    except Exception as e:
        log.warning(f"Spelling correction failed: {e}")
        return ""


def _query_variants(question: str, lang: str) -> List[str]:
    """
    Build robust retrieval variants:
    - original query
    - normalized Arabic / lowercase English
    - typo-corrected query
    - Arabic <-> English translations
    - rephrases
    """
    q = _clean(question)
    if not q:
        return []

    variants: List[str] = []

    def add(x: str):
        x = _clean(x)
        if x and x not in variants:
            variants.append(x)

    detected = detect_language(q)

    add(q)
    add(_normalize(q))

    if detected == "ar" or _is_mixed_language(q):
        add(_normalize_arabic(q))
        add(_loose_arabic(q))
    else:
        add(q.lower())
        add(_loose_english(q))

    fixed = _fix_query_spelling(q, detected)
    if fixed:
        add(fixed)
        add(_normalize(fixed))
        if detect_language(fixed) == "ar":
            add(_normalize_arabic(fixed))
            add(_loose_arabic(fixed))
        else:
            add(fixed.lower())
            add(_loose_english(fixed))

    # Always try Arabic -> English
    tr_en = _translate(q, "en")
    if tr_en:
        add(tr_en)
        add(tr_en.lower())
        add(_loose_english(tr_en))

        fixed_en = _fix_query_spelling(tr_en, "en")
        if fixed_en:
            add(fixed_en)
            add(fixed_en.lower())
            add(_loose_english(fixed_en))

        rp_en = _rephrase(tr_en, "en")
        if rp_en:
            add(rp_en)
            add(rp_en.lower())
            add(_loose_english(rp_en))

    # Always try English -> Arabic
    tr_ar = _translate(q, "ar")
    if tr_ar:
        add(tr_ar)
        add(_normalize_arabic(tr_ar))
        add(_loose_arabic(tr_ar))

        fixed_ar = _fix_query_spelling(tr_ar, "ar")
        if fixed_ar:
            add(fixed_ar)
            add(_normalize_arabic(fixed_ar))
            add(_loose_arabic(fixed_ar))

        rp_ar = _rephrase(tr_ar, "ar")
        if rp_ar:
            add(rp_ar)
            add(_normalize_arabic(rp_ar))
            add(_loose_arabic(rp_ar))

    return variants[:18]


# ── Lexical scoring ────────────────────────────────────────────────────────────

def _lex_score(query: str, doc_text: str, lang: str) -> float:
    kws = _keywords(query, lang)
    if not kws:
        return 0.0
    doc_norm = _normalize(doc_text)
    uni_score = sum(1 for kw in kws if kw in doc_norm) / max(len(kws), 1)
    bigs = _ngrams(query, 2)
    bigram_score = (sum(1 for bg in bigs if bg in doc_norm) / max(len(bigs), 1)) * 1.5 if bigs else 0.0
    return min(uni_score + bigram_score, 1.0)


# ── Reranking ────────────────────────────────────────────────────────────────

def _rerank(variants: List[str], docs: List[Document], top_n: Optional[int] = None) -> Tuple[List[Document], List[dict]]:
    """
    Rerank without throwing away vector-search results.
    Important for Arabic <-> English cross-language questions and typo-heavy queries.
    """
    top_n = top_n or settings.RERANK_TOP_N
    scored = []

    for idx, d in enumerate(docs):
        content = _clean(d.page_content)
        if not _is_meaningful(content):
            continue

        max_lex = 0.0

        for qv in variants:
            q_lang = detect_language(qv)
            s = _lex_score(qv, content, q_lang)

            if q_lang == "ar":
                loose_q = _loose_arabic(qv)
                loose_doc = _loose_arabic(content)
                kws = [w for w in loose_q.split() if len(w) > 2]
                if kws:
                    loose_score = sum(1 for kw in kws if kw in loose_doc) / len(kws)
                    s = max(s, loose_score)
            else:
                loose_q = _loose_english(qv)
                loose_doc = _loose_english(content)
                kws = [w for w in loose_q.split() if len(w) > 2]
                if kws:
                    loose_score = sum(1 for kw in kws if kw in loose_doc) / len(kws)
                    s = max(s, loose_score)

            max_lex = max(max_lex, s)

        # Keep vector-search ordering as fallback even if lexical score is low.
        final_score = max_lex - idx * 0.001

        scored.append((final_score, d, {
            "source": d.metadata.get("source", "?"),
            "page": d.metadata.get("page", 0),
            "score": round(final_score, 4),
            "preview": content[:120].replace("\n", " "),
        }))

    scored.sort(key=lambda x: x[0], reverse=True)

    ranked = [d for _, d, _ in scored]
    debugs = [dbg for _, _, dbg in scored]

    return ranked[:top_n], debugs[:top_n]


def _retrieve(question: str, lang: str, k: Optional[int] = None, top_n: Optional[int] = None) -> Tuple[List[Document], str]:
    """
    Retrieve + rerank documents for a question.

    `k` overrides how many candidates are pulled per query variant
    (defaults to settings.RETRIEVER_K); `top_n` overrides how many
    reranked documents are returned (defaults to settings.RERANK_TOP_N).
    Used internally by `ask_question` and exposed via `retrieve()` for
    the agent's retrieve tool.
    """
    vdb = _get_vector_db()
    retriever = vdb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k or settings.RETRIEVER_K},
    )

    variants = _query_variants(question, lang)

    all_docs: List[Document] = []

    for q in variants:
        try:
            docs = retriever.invoke(q)
            all_docs.extend(docs)
        except Exception as e:
            log.error(f"Retrieval error for '{q}': {e}")

    # Fallback for very typo-heavy or short queries
    if not all_docs:
        simplified = " ".join(
            re.findall(r"[\u0600-\u06FFa-zA-Z]{2,}", _normalize(question))[:8]
        )
        if simplified:
            try:
                all_docs.extend(retriever.invoke(simplified))
            except Exception as e:
                log.error(f"Fallback retrieval error: {e}")

    all_docs = _deduplicate_retrieved(all_docs)

    if not all_docs:
        return [], "no docs retrieved"

    ranked, debugs = _rerank(variants, all_docs, top_n=top_n)

    if not debugs:
        return [], "no docs retrieted after rerank"

    top_score = debugs[0]["score"]

    # Coarse, defense-in-depth filter only: this lexical overlap score is a
    # heuristic and NOT a reliable topic-relevance classifier on its own
    # (short/loosely-worded but genuinely relevant questions can score
    # similarly to unrelated ones). It only catches near-zero-overlap
    # queries. The real grounding guarantee is the "answer only if the
    # context specifically covers the question, else say unavailable" rule
    # in build_prompt() — this is just a cheap first line of defense.
    if top_score < settings.CONFIDENCE_THRESHOLD:
        debug_str = "\n".join(
            f"Rank {i+1}: {d['source']} p{d['page']} score={d['score']} | {d['preview'][:80]}"
            for i, d in enumerate(debugs)
        )
        return [], f"{debug_str}\n[top score {top_score} below CONFIDENCE_THRESHOLD={settings.CONFIDENCE_THRESHOLD} — treating as no relevant match]"

    debug_str = "\n".join(
        f"Rank {i+1}: {d['source']} p{d['page']} score={d['score']} | {d['preview'][:80]}"
        for i, d in enumerate(debugs)
    )

    return ranked, debug_str


# ── Prompt & cleanup ────────────────────────────────────────────────────────────

def build_prompt(context: str, question: str, lang: str) -> str:
    if lang == "ar":
        return f"""أنت نظام استخراج معلومات متقدم. مهمتك: استخرج إجابة كاملة ومفصّلة من السياق المقدم فقط.

**قواعد الإجابة — يجب اتباعها بدقة:**

1. اقرأ السياق كاملاً قبل الكتابة.
2. السؤال قد يكون بالعربية والسياق بالإنجليزية أو العكس؛ افهم المعنى بين اللغتين.
3. السؤال قد يحتوي على أخطاء إملائية أو حروف ناقصة أو كلمات عامية؛ حاول فهم المقصود من السياق.
4. أجب من السياق فقط إذا كان يحتوي بشكل مباشر ومحدد على المعلومة المطلوبة في السؤال — وليس مجرد موضوع مشابه أو قريب.
5. إذا كان السياق لا يحتوي على المعلومة المحددة المطلوبة، يجب أن تقول "المعلومة غير موجودة في الملفات المرفوعة." لا تجب إجابة جزئية بالاعتماد على فقرة قريبة الموضوع فقط، ولا تسد الفجوة بمعرفتك الخاصة — حتى لو كنت متأكدًا أنها معلومة صحيحة في الواقع. هذا النظام يجب أن يجيب حصريًا من المستندات المرفوعة، ولا شيء غير ذلك.
6. لا تستخدم أي معرفة خارجية خارج السياق تحت أي ظرف، حتى للأسئلة التي تستطيع الإجابة عنها بسهولة بنفسك (حقائق عامة، تعريفات، أحداث جارية، إلخ). إذا لم تكن المعلومة في السياق، فلا تضعها في الإجابة.
7. لا تبدأ بعبارات مثل "بناءً على السياق" أو "وفقاً للمعلومات".
8. لا تكرر السؤال في الإجابة.
9. اذكر الأرقام والتواريخ والأسماء كما وردت في السياق.
9ب. حافظ على المعادلات والصيغ والوحدات والمصطلحات التقنية كما هي بالضبط في السياق؛ لا تعيد صياغتها أو تقرّبها أو تبسّطها.
9ج. لا تخترع رقمًا أو اسمًا أو مصطلحًا أو حقيقة غير موجودة في السياق، حتى لو كان ذلك لسد فجوة أو لجعل الإجابة تبدو مكتملة.
10. الإجابة بالعربية الواضحة، مع الحفاظ على المصطلحات الإنجليزية المهمة كما هي.
11. بعد الإجابة، قدم مثالًا بسيطًا إذا كان مناسبًا.
12. استخدم تنسيق:
    - الشرح:
    - المثال:

**السياق:**
{context}

**السؤال:**
{question}

**الإجابة الكاملة:**"""

    return f"""You are an advanced information extraction system. Your task is to extract a complete, well-structured answer from the provided context only.

**Answering rules — follow precisely:**

1. Read the entire context before writing.
2. The question and context may be in different languages. Understand the meaning across Arabic and English.
3. The question may contain spelling mistakes, missing letters, dialect words, or mixed Arabic/English terms. Infer the intended meaning from the context.
4. Answer using the context ONLY if it specifically and directly contains the information the question is asking for — not merely a related or similar topic.
5. If the context does not contain the specific information requested, you MUST say "The information is not available in the uploaded files." Do not partially answer using a superficially related passage, and do not fill the gap with your own knowledge — even if you are confident it is correct real-world information. This system must answer strictly from the uploaded documents, nothing else.
6. Do not use external knowledge outside the context, under any circumstance, even for questions you could easily answer yourself (general facts, definitions, current events, etc.). If it isn't in the context, it isn't in the answer.
7. Do not open with "Based on the context" or "According to the information".
8. Do not repeat the question.
9. State numbers, dates, and names exactly as they appear in the context.
9b. Preserve equations, formulas, units, and technical terms exactly as written in the context — do not paraphrase, round, or simplify them.
9c. Never invent a number, name, term, or fact that is not present in the context, even to fill a gap or sound complete.
10. Answer in clear English, preserving important Arabic or English technical terms when needed.
11. After the answer, provide a simple example if useful.
12. Use format:
   - Explanation:
   - Example:

**Context:**
{context}

**Question:**
{question}

**Complete Answer:**"""


def build_prompt_with_memory(context: str, question: str, lang: str, memory: str = "") -> str:
    """Same as `build_prompt`, with optional conversation memory prepended."""
    base = build_prompt(context, question, lang)
    if not memory:
        return base
    header = "**Conversation memory (for context only, do not repeat it):**" if lang != "ar" else "**ذاكرة المحادثة (للسياق فقط، لا تكررها):**"
    return f"{header}\n{memory}\n\n{base}"


def _clean_answer(text: str, lang: str) -> str:
    text = _clean(text)
    banned = [
        "Sources:", "Source:", "المصادر:", "المصدر:",
        "Based on the context,", "According to the context,",
        "بناءً على السياق،", "وفقاً للمعلومات المتاحة،",
    ]
    lines, seen = [], []
    for line in text.splitlines():
        for b in banned:
            if line.strip().startswith(b):
                line = line.replace(b, "", 1).strip()
                break
        key = _normalize(line)
        if key and key in seen:
            continue
        lines.append(line)
        if key:
            seen.append(key)
            if len(seen) > 10:
                seen.pop(0)
    result = "\n".join(lines).strip()
    if not result:
        return ("المعلومة غير موجودة في الملفات المرفوعة." if lang == "ar"
                else "The information is not available in the uploaded files.")
    return result


def _build_sources(docs: List[Document], lang: str) -> str:
    if not docs:
        return ""
    counts: Counter = Counter()
    pages: dict = {}
    for d in docs:
        src = os.path.basename(d.metadata.get("source", "?"))
        pg = d.metadata.get("page", 0)
        counts[src] += 1
        pages.setdefault(src, set()).add(pg + 1 if isinstance(pg, int) else pg)
    parts = []
    for src, _ in counts.most_common(3):
        pg_str = ", ".join(map(str, sorted(list(pages.get(src, [])))[:3]))
        parts.append(f"{src} (p. {pg_str})" if pg_str else src)
    prefix = "المصادر: " if lang == "ar" else "Sources: "
    return prefix + " | ".join(parts)


# ── File registry ──────────────────────────────────────────────────────────────

def _load_registry() -> dict:
    if os.path.isfile(settings.PROCESSED_FILES_REGISTRY):
        try:
            with open(settings.PROCESSED_FILES_REGISTRY, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_registry(reg: dict):
    try:
        with open(settings.PROCESSED_FILES_REGISTRY, "w", encoding="utf-8") as f:
            json.dump(reg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"Registry save error: {e}")


def _file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_filename(filename: str) -> str:
    name = os.path.basename(filename or "unknown")
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    return name.strip() or "unknown"


_CONTENT_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc": "application/msword",
    "txt": "text/plain",
    "markdown": "text/markdown",
    "json": "application/json",
    "image": "application/octet-stream",
}


def _save_uploaded_file(filename: str, data: bytes, fhash: str) -> str:
    """
    Upload the original file bytes to the MinIO uploads bucket instead of
    local disk. Returns the MinIO object key (stored in the registry as
    "stored_path" for backward compatibility with existing callers/fields).
    """
    safe_name = _safe_filename(filename)
    stem, ext = os.path.splitext(safe_name)

    object_name = f"{stem}_{fhash[:10]}{ext}"
    content_type = _CONTENT_TYPES.get(_get_file_type(filename), "application/octet-stream")

    storage_service.upload_bytes(
        settings.MINIO_BUCKET_UPLOADS, object_name, data, content_type=content_type
    )

    return object_name


def get_stored_file_bytes(object_name: str) -> bytes:
    """Download a previously uploaded file's raw bytes from MinIO."""
    return storage_service.download_bytes(settings.MINIO_BUCKET_UPLOADS, object_name)


def list_stored_files() -> list[dict]:
    registry = _load_registry()
    files = []

    for _, info in registry.items():
        object_name = info.get("stored_path")
        files.append({
            "filename": info.get("filename"),
            "stored_path": object_name,
            "file_type": info.get("file_type"),
            "chunks": info.get("chunks", 0),
            "processed_at": info.get("processed_at"),
            "download_url": storage_service.presigned_url(
                settings.MINIO_BUCKET_UPLOADS, object_name
            ) if object_name else None,
        })

    files.sort(key=lambda x: x.get("processed_at") or "", reverse=True)
    return files


def find_registry_entry(filename: str) -> Optional[dict]:
    """Look up a stored file's registry entry (incl. its MinIO object key)
    by original filename. Used by the report-generation endpoint."""
    registry = _load_registry()
    for info in registry.values():
        if info.get("filename") == filename:
            return info
    return None


def get_document_pages(filename: str) -> List[dict]:
    """
    Re-download a previously ingested file from MinIO and re-extract its
    full text, page by page: [{"page": <int>, "text": "..."}, ...].
    Used by report_service to build a comprehensive report while keeping
    track of which page each fact/section came from.
    """
    entry = find_registry_entry(filename)
    if not entry:
        raise FileNotFoundError(f"No stored file found for '{filename}'.")

    data = get_stored_file_bytes(entry["stored_path"])
    docs = _load_document_from_bytes(filename, data)

    pages: dict[int, list[str]] = {}
    for d in docs:
        page = d.metadata.get("page", 0)
        pages.setdefault(page, []).append(d.page_content)

    return [
        {"page": page + 1, "text": "\n".join(texts).strip()}
        for page, texts in sorted(pages.items())
        if "\n".join(texts).strip()
    ]


def get_document_full_text(filename: str) -> str:
    """Convenience wrapper over `get_document_pages` for callers that just
    want the whole document as one string, without page boundaries."""
    return "\n\n".join(p["text"] for p in get_document_pages(filename))


def _get_file_type(filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    return {"pdf": "pdf", "docx": "docx", "doc": "doc", "txt": "txt",
            "md": "markdown", "json": "json", "jpg": "image",
            "jpeg": "image", "png": "image"}.get(ext, "unknown")


# ── Document loading from bytes ────────────────────────────────────────────────

def _load_document_from_bytes(filename: str, data: bytes) -> List[Document]:
    """Load, OCR if needed, and enrich a document from its raw bytes."""
    ext = filename.lower().rsplit(".", 1)[-1]
    filetype = _get_file_type(filename)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta_base = {"source": filename, "file_type": filetype, "page": 0, "timestamp": ts}

    raw_docs: List[Document] = []

    try:
        if ext == "pdf":
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                tf.write(data)
                tmp_path = tf.name
            try:
                loader = PyPDFLoader(tmp_path)
                raw_docs = loader.load()
                text_body = "".join(d.page_content for d in raw_docs).strip()
                if settings.ENABLE_PDF_OCR_FALLBACK and len(text_body) < 20:
                    ocr_text = perform_ocr_pdf_bytes(data)
                    if _clean(ocr_text):
                        raw_docs = [Document(page_content=ocr_text, metadata={**meta_base, "ocr_fallback": True})]
                    else:
                        raw_docs = []
                else:
                    for i, d in enumerate(raw_docs):
                        d.metadata = {**meta_base, "page": d.metadata.get("page", i)}
            finally:
                os.unlink(tmp_path)

        elif ext in {"docx", "doc"}:
            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tf:
                tf.write(data)
                tmp_path = tf.name
            try:
                raw_docs = Docx2txtLoader(tmp_path).load()
                for d in raw_docs:
                    d.metadata = {**meta_base}
            finally:
                os.unlink(tmp_path)

        elif ext in {"txt", "md"}:
            text = data.decode("utf-8", errors="replace").strip()
            raw_docs = [Document(page_content=text, metadata={**meta_base})]

        elif ext == "json":
            try:
                obj = json.loads(data.decode("utf-8", errors="replace"))
                text = json.dumps(obj, ensure_ascii=False, indent=2)
            except Exception:
                text = data.decode("utf-8", errors="replace")
            raw_docs = [Document(page_content=text, metadata={**meta_base})]

        elif ext in {"jpg", "jpeg", "png", "tiff", "bmp", "webp"}:
            text = perform_ocr_image_bytes(data)
            if _clean(text):
                raw_docs = [Document(page_content=text, metadata={**meta_base, "ocr": True})]

    except Exception as e:
        log.error(f"load_document error ({filename}): {e}")

    enriched = []
    for d in raw_docs:
        content = _enrich(d.page_content)
        if content:
            d.page_content = content
            enriched.append(d)

    log.info(f"Loaded '{filename}' -> {len(enriched)} docs")
    return enriched


# ── Public API: ingestion ───────────────────────────────────────────────────────

def update_db_files(files: List[dict[str, Any]]) -> int:
    """
    Ingest a list of {'filename': str, 'data': bytes} dicts into Qdrant.
    Also uploads the original file bytes to MinIO (settings.MINIO_BUCKET_UPLOADS).
    Returns total number of chunks added.
    """
    registry = _load_registry()
    new_files = []
    skipped = []

    for f in files:
        filename = f.get("filename") or "unknown"
        data = f.get("data") or b""

        if not data:
            log.warning(f"Skip empty file: {filename}")
            continue

        h = _file_hash(data)

        if h in registry:
            skipped.append(filename)
            log.info(f"Skip already processed: {filename}")
        else:
            new_files.append((filename, data, h))

    if not new_files:
        log.info(f"No new files — skipped: {skipped}")
        return 0

    all_docs: List[Document] = []
    per_file_info: dict = {}

    for filename, data, fhash in new_files:
        stored_path = _save_uploaded_file(filename, data, fhash)

        docs = _load_document_from_bytes(filename, data)

        for d in docs:
            d.metadata["stored_path"] = stored_path

        all_docs.extend(docs)

        per_file_info[filename] = {
            "hash": fhash,
            "stored_path": stored_path,
            "docs_count": len(docs),
        }

    all_docs = _deduplicate(all_docs)

    if not all_docs:
        log.warning("No valid documents extracted from uploaded files.")
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n\n", "\n\n", "\n", ".", " ", ""],
    )

    chunks = [
        c for c in splitter.split_documents(all_docs)
        if _is_meaningful(c.page_content)
    ]

    if not chunks:
        log.warning("No meaningful chunks generated.")
        return 0

    cps: Counter = Counter(c.metadata.get("source", "?") for c in chunks)

    idx_map: Counter = Counter()

    for chunk in chunks:
        src = chunk.metadata.get("source", "?")
        chunk.metadata["chunk_index"] = idx_map[src]
        chunk.metadata["total_chunks"] = cps[src]
        idx_map[src] += 1

    ensure_collection(embeddings)

    vdb = _get_vector_db()
    vdb.add_documents(chunks)

    _refresh_retriever()

    for filename, info in per_file_info.items():
        fhash = info["hash"]

        registry[fhash] = {
            "filename": filename,
            "stored_path": info["stored_path"],
            "file_type": _get_file_type(filename),
            "chunks": cps.get(filename, 0),
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    _save_registry(registry)

    log.info(
        f"Added {len(chunks)} chunks from {len(new_files)} new file(s). "
        f"Skipped: {skipped}"
    )

    return len(chunks)


# ── Public API: direct question answering (non-agent /chat path) ───────────────

def ask_question(query: str, lang: str = "auto") -> dict:
    """
    Answer a question using RAG directly (single retrieval + single generation).
    Returns {"answer": str, "sources": str}.
    """
    if not is_ready():
        return {
            "answer": "⚠️ Database is empty. Please upload files first.",
            "sources": "",
        }

    detected_lang = detect_language(query) if lang == "auto" else lang
    docs, debug = _retrieve(query, detected_lang)
    log.debug(f"Retrieval:\n{debug}")

    if not docs:
        no_info = ("المعلومة غير موجودة في الملفات المرفوعة."
                   if detected_lang == "ar"
                   else "The information is not available in the uploaded files.")
        return {"answer": no_info, "sources": ""}

    context = _build_context(docs)

    prompt = build_prompt(context, query, detected_lang)
    try:
        t0 = time.time()
        answer = str(llm.invoke(prompt))
        log.info(f"LLM answered in {time.time()-t0:.2f}s")
        answer = _clean_answer(answer, detected_lang)
    except Exception as e:
        log.error(f"LLM error: {e}")
        answer = f"Error generating answer: {e}"

    return {"answer": answer, "sources": _build_sources(docs, detected_lang)}


def _build_context(docs: List[Document]) -> str:
    context_parts = [
        f"[Chunk {i+1} | {d.metadata.get('source','?')} | page {d.metadata.get('page',0)}]\n{d.page_content}"
        for i, d in enumerate(docs)
    ]
    return "\n\n---\n\n".join(context_parts)


# ── Public API used by the agent's tools ────────────────────────────────────────
#
# These functions wrap the pipeline above in a stable, agent-friendly shape
# (plain dicts instead of langchain Document objects) so the agent package
# never has to know about langchain/Qdrant internals.

def retrieve(question: str, lang: str = "auto", top_k: int = 5) -> List[dict]:
    """
    Retrieve relevant chunks for a question. Used by the agent's retrieve tool.

    Returns a list of dicts: {"id", "text", "metadata": {...}}
    """
    if not is_ready():
        return []

    detected_lang = detect_language(question) if lang == "auto" else lang
    docs, debug = _retrieve(question, detected_lang, k=max(top_k, settings.RETRIEVER_K), top_n=top_k)
    log.debug(f"[agent] Retrieval for '{question}':\n{debug}")

    results = []
    for d in docs:
        source = d.metadata.get("source", "?")
        page = d.metadata.get("page", 0)
        chunk_index = d.metadata.get("chunk_index", 0)
        chunk_id = f"{source}::{page}::{chunk_index}::{hashlib.md5(d.page_content.encode('utf-8')).hexdigest()[:8]}"
        results.append({
            "id": chunk_id,
            "text": d.page_content,
            "metadata": {
                "source": source,
                "title": source,
                "document_type": d.metadata.get("file_type", "unknown"),
                "page": page,
                "chunk_index": chunk_index,
            },
        })
    return results


def generate_answer(question: str, documents: List[dict], lang: str = "auto", memory: str = "") -> str:
    """
    Generate a final answer from agent-collected documents (+ optional
    conversation memory). Used by the agent's generate tool.
    """
    detected_lang = detect_language(question) if lang == "auto" else lang

    if not documents:
        return ("لا توجد مستندات كافية للإجابة على هذا السؤال."
                if detected_lang == "ar"
                else "No relevant documents were found to answer this question.")

    context_parts = [
        f"[Chunk {i+1} | {d['metadata'].get('source','?')} | page {d['metadata'].get('page',0)}]\n{d['text']}"
        for i, d in enumerate(documents)
    ]
    context = "\n\n---\n\n".join(context_parts)

    prompt = build_prompt_with_memory(context, question, detected_lang, memory)

    try:
        answer = str(llm.invoke(prompt))
        return _clean_answer(answer, detected_lang)
    except Exception as e:
        log.error(f"[agent] generate_answer error: {e}")
        return f"Error generating answer: {e}"


def generate_answer_stream(question: str, documents: List[dict], lang: str = "auto", memory: str = ""):
    """
    Streaming counterpart of `generate_answer`: yields text chunks as the
    Groq API produces them, instead of returning the full string at once.
    Used by the agent's `run_stream` for the /ws/chat WebSocket endpoint.
    """
    detected_lang = detect_language(question) if lang == "auto" else lang

    if not documents:
        yield ("لا توجد مستندات كافية للإجابة على هذا السؤال."
               if detected_lang == "ar"
               else "No relevant documents were found to answer this question.")
        return

    context_parts = [
        f"[Chunk {i+1} | {d['metadata'].get('source','?')} | page {d['metadata'].get('page',0)}]\n{d['text']}"
        for i, d in enumerate(documents)
    ]
    context = "\n\n---\n\n".join(context_parts)

    prompt = build_prompt_with_memory(context, question, detected_lang, memory)

    try:
        for chunk in llm.stream(prompt):
            yield chunk
    except Exception as e:
        log.error(f"[agent] generate_answer_stream error: {e}")
        yield f"Error generating answer: {e}"


def _memory_only_prompt(question: str, memory: str, lang: str) -> str:
    """
    Prompt for the 'respond' tool / no-document fallback: strictly scoped
    to greetings, small talk, and questions about the conversation itself.
    Explicitly forbidden from answering general-knowledge/factual questions
    from the model's own training data — this system must only ever answer
    from the uploaded documents.
    """
    if lang == "ar":
        return f"""أنت وحدة رد صغيرة داخل نظام سؤال وجواب عن مستندات مرفوعة. هذا النظام يجيب حصريًا من المستندات المرفوعة — لا من معرفتك العامة.

استخدم هذا الرد فقط لأحد الحالات التالية:
- تحية أو مجاملة أو شكر أو وداع.
- سؤال عن المحادثة نفسها (مثل "ايه اللي سألتك عليه قبل كده؟") ويمكن الإجابة عليه من ذاكرة المحادثة أدناه فقط.

إذا كانت رسالة المستخدم تطلب أي معلومة أو حقيقة أو تفسير (حتى لو كانت معلومة عامة تعرفها جيدًا، مثل عاصمة دولة أو حدث تاريخي أو تعريف علمي) ولم تكن هذه المعلومة مذكورة حرفيًا في ذاكرة المحادثة أدناه، فلا تجب عليها من معرفتك. بدلاً من ذلك، رد بأدب أن هذا النظام يجيب فقط من المستندات المرفوعة، واقترح أن يسأل عن محتوى المستندات.

ذاكرة المحادثة:
{memory or "لا توجد ذاكرة سابقة."}

رسالة المستخدم:
{question}

الرد:"""

    return f"""You are a small response module inside a document Q&A system. This system answers strictly from uploaded documents — never from your own general knowledge.

Only use this reply for one of these cases:
- A greeting, thanks, small talk, or farewell.
- A question about the conversation itself (e.g. "what did I just ask you?") that the conversation memory below can answer.

If the user's message asks for any fact, information, or explanation — even something you personally know well, like a country's capital, a historical event, or a scientific definition — and that information is NOT explicitly present in the conversation memory below, do not answer it from your own knowledge. Instead, politely say this system only answers questions about the uploaded documents, and invite them to ask about the documents.

Conversation memory:
{memory or "No prior memory."}

User message:
{question}

Reply:"""


def answer_from_memory_stream(question: str, memory: str, lang: str = "auto"):
    """Streaming counterpart of `answer_from_memory`."""
    detected_lang = detect_language(question) if lang == "auto" else lang
    prompt = _memory_only_prompt(question, memory, detected_lang)

    try:
        for chunk in llm.stream(prompt):
            yield chunk
    except Exception as e:
        log.error(f"[agent] answer_from_memory_stream error: {e}")
        yield f"Error generating answer: {e}"


def summarize_stream(documents: List[dict], lang: str = "en"):
    """Streaming counterpart of `summarize`."""
    if not documents:
        yield "لا توجد مستندات لتلخيصها." if lang == "ar" else "No documents found to summarize."
        return

    document_text = "\n\n".join(d["text"] for d in documents)
    prompt = (
        f"لخّص المستندات التالية بإيجاز ووضوح، بالاعتماد فقط على ما ورد فيها حرفيًا، دون إضافة أي معلومة من خارجها:\n\n{document_text}"
        if lang == "ar"
        else f"Summarize the following documents clearly and concisely, using only what is stated in them — do not add any information from outside them:\n\n{document_text}"
    )

    try:
        for chunk in llm.stream(prompt):
            yield chunk
    except Exception as e:
        log.error(f"[agent] summarize_stream error: {e}")
        yield f"Error generating summary: {e}"


def compare_stream(question: str, documents: List[dict], lang: str = "en"):
    """Streaming counterpart of `compare`."""
    if not documents:
        yield "لا توجد مستندات كافية للمقارنة." if lang == "ar" else "No documents found to compare."
        return

    document_text = "\n\n".join(d["text"] for d in documents)
    prompt = (
        f"باستخدام المستندات التالية حصريًا (لا تستخدم أي معرفة خارجية)، أجب عن طلب المقارنة. إذا كانت المستندات لا تحتوي على معلومات كافية للمقارنة المطلوبة، قل ذلك صراحة:\n\nالسؤال:\n{question}\n\nالمستندات:\n\n{document_text}"
        if lang == "ar"
        else f"Using only the following documents (no outside knowledge), answer this comparison request. If the documents do not contain enough information for the requested comparison, say so explicitly:\n\nQuestion:\n{question}\n\nDocuments:\n\n{document_text}"
    )

    try:
        for chunk in llm.stream(prompt):
            yield chunk
    except Exception as e:
        log.error(f"[agent] compare_stream error: {e}")
        yield f"Error generating comparison: {e}"


def answer_from_memory(question: str, memory: str, lang: str = "auto") -> str:
    """
    Answer a question using only conversation memory (no document retrieval).
    Strictly scoped to greetings/small talk/meta-conversation — see
    `_memory_only_prompt` for the full rule set. Used by the 'respond' tool
    and by 'generate' when no documents were retrieved but memory exists.
    """
    detected_lang = detect_language(question) if lang == "auto" else lang
    prompt = _memory_only_prompt(question, memory, detected_lang)

    try:
        answer = str(llm.invoke(prompt))
        return _clean_answer(answer, detected_lang)
    except Exception as e:
        log.error(f"[agent] answer_from_memory error: {e}")
        return f"Error generating answer: {e}"


def summarize(documents: List[dict], lang: str = "en") -> str:
    """Summarize a set of agent-collected documents. Used by the summarize tool."""
    if not documents:
        return "لا توجد مستندات لتلخيصها." if lang == "ar" else "No documents found to summarize."

    document_text = "\n\n".join(d["text"] for d in documents)

    if lang == "ar":
        prompt = f"لخّص المستندات التالية بإيجاز ووضوح، بالاعتماد فقط على ما ورد فيها حرفيًا، دون إضافة أي معلومة من خارجها:\n\n{document_text}"
    else:
        prompt = f"Summarize the following documents clearly and concisely, using only what is stated in them — do not add any information from outside them:\n\n{document_text}"

    try:
        return _clean(str(llm.invoke(prompt)))
    except Exception as e:
        log.error(f"[agent] summarize error: {e}")
        return f"Error generating summary: {e}"


def compare(question: str, documents: List[dict], lang: str = "en") -> str:
    """Compare information across agent-collected documents. Used by the compare tool."""
    if not documents:
        return "لا توجد مستندات كافية للمقارنة." if lang == "ar" else "No documents found to compare."

    document_text = "\n\n".join(d["text"] for d in documents)

    if lang == "ar":
        prompt = f"باستخدام المستندات التالية حصريًا (لا تستخدم أي معرفة خارجية)، أجب عن طلب المقارنة. إذا كانت المستندات لا تحتوي على معلومات كافية للمقارنة المطلوبة، قل ذلك صراحة:\n\nالسؤال:\n{question}\n\nالمستندات:\n\n{document_text}"
    else:
        prompt = f"Using only the following documents (no outside knowledge), answer this comparison request. If the documents do not contain enough information for the requested comparison, say so explicitly:\n\nQuestion:\n{question}\n\nDocuments:\n\n{document_text}"

    try:
        return _clean(str(llm.invoke(prompt)))
    except Exception as e:
        log.error(f"[agent] compare error: {e}")
        return f"Error generating comparison: {e}"


def build_sources_from_dicts(documents: List[dict], lang: str = "en") -> str:
    """Build a human-readable sources string from agent document dicts."""
    if not documents:
        return ""
    counts: Counter = Counter()
    pages: dict = {}
    for d in documents:
        meta = d.get("metadata", {})
        src = os.path.basename(meta.get("source", "?"))
        pg = meta.get("page", 0)
        counts[src] += 1
        pages.setdefault(src, set()).add(pg + 1 if isinstance(pg, int) else pg)
    parts = []
    for src, _ in counts.most_common(3):
        pg_str = ", ".join(map(str, sorted(list(pages.get(src, [])))[:3]))
        parts.append(f"{src} (p. {pg_str})" if pg_str else src)
    prefix = "المصادر: " if lang == "ar" else "Sources: "
    return prefix + " | ".join(parts)
