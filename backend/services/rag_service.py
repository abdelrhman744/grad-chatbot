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
import time
import uuid
from collections import Counter
from functools import lru_cache
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http import models as qdrant_models

from config import settings
from loaders.registry import (
    get_file_type as _get_file_type,
    get_content_type,
    load_document_from_bytes as _dispatch_load,
)
from services.db_service import get_client, get_collection_name, ensure_collection, with_retries
from services.embeddings_provider import get_embeddings
from services.llm_provider import GroqLLM, get_llm as _get_shared_llm, get_agent_llm
from services import storage_service
from utils import timing
from utils.prompt_safety import wrap_untrusted_context

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
    """Call once at startup to attach to an existing Qdrant collection.
    Retries a bounded number of times (Qdrant may still be starting up),
    but never raises — if it still can't connect, the app starts anyway
    and retrieval simply reports not-ready until a later call succeeds."""
    global _vector_db

    def _attach() -> int:
        client = get_client()
        client.get_collection(get_collection_name())
        return client.get_collection(get_collection_name()).points_count

    try:
        count = with_retries(_attach, what="load_existing_db")
        _vector_db = QdrantVectorStore(
            client=get_client(),
            collection_name=get_collection_name(),
            embedding=embeddings,
        )
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


# ── Document isolation ──────────────────────────────────────────────────────
# All conversations still share ONE physical Qdrant collection (document-
# level storage isolation, e.g. one collection/namespace per conversation,
# is explicitly out of scope for this change) — isolation is enforced at
# query time instead, via a metadata filter on every vector search, plus
# tagging every indexed chunk with the conversation_id that uploaded it at
# ingestion time. See `_conversation_filter`, `update_db_files`, `_retrieve`.

def _conversation_filter(conversation_id: str) -> qdrant_models.Filter:
    """
    Qdrant filter restricting a search/delete to exactly one
    conversation_id's chunks. `langchain_qdrant.QdrantVectorStore` stores
    each Document's metadata dict under the payload key "metadata" (its
    default `metadata_payload_key` — verified against the installed
    langchain-qdrant version), so the field path is "metadata.conversation_id",
    not a top-level "conversation_id" key.
    """
    return qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="metadata.conversation_id",
                match=qdrant_models.MatchValue(value=conversation_id),
            )
        ]
    )


def _document_filter(document_id: str, conversation_id: str) -> qdrant_models.Filter:
    """
    Qdrant filter restricting a search/delete to exactly one document_id's
    chunks — AND'd with conversation_id (same field-path reasoning as
    `_conversation_filter`) so a re-index or delete can never cross a
    conversation boundary even if a document_id were guessed or leaked:
    both conditions must match, not just document_id alone. Used by
    `reindex_document` and `delete_document`.
    """
    return qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="metadata.document_id",
                match=qdrant_models.MatchValue(value=document_id),
            ),
            qdrant_models.FieldCondition(
                key="metadata.conversation_id",
                match=qdrant_models.MatchValue(value=conversation_id),
            ),
        ]
    )


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

def _run_concurrent(tasks: List) -> List:
    """
    Run independent, blocking callables (LLM calls) concurrently on a small
    thread pool and return their results in the same order. Several of the
    query-variant reformulation calls below are logically independent (they
    don't depend on each other's output) but used to run one-by-one; this
    turns N sequential Groq round-trips into a single wall-clock wave.

    Tasks run via `timing.run_concurrent_ctx` (not a bare ThreadPoolExecutor)
    so that stage()/substage() calls made inside each task still land on the
    current request's profiler — see utils/timing.py.
    """
    return timing.run_concurrent_ctx(tasks)


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


# Single-token queries that look like a bare identifier — an IP address,
# hostname, URL, error code with a numeric part, version string, etc. —
# have nothing for an LLM to usefully translate or paraphrase: there's no
# natural-language content to fix typos in or find synonyms for, and
# "translating" "192.168.1.1" is meaningless. Skipping the LLM call for
# these is safe — local normalization (_normalize/_loose_*) still runs on
# them exactly as for any other query, so they keep variant coverage, just
# not synonym/translation variants that wouldn't help.
#
# Deliberately requires a digit or a '.'/'/'/':'  — NOT just "single token,
# no spaces" — so a plain single-word acronym/topic query (e.g. "PEAS",
# "blockchain") still gets the full rewrite/translate pass. A bare word
# like that is exactly the case PROFILING.md's cross-language investigation
# tested (an Arabic-only query needing translation to match an
# English-only document, and vice versa) — skipping translation for it
# would silently reintroduce that exact recall gap. An identifier-like
# error code with no digit/separator at all (e.g. "ERR_CONN_TIMEOUT")
# isn't matched either — it still gets one (cheap, fast-model) LLM call
# rather than risk under-matching a real word.
_CODE_LIKE_RE = re.compile(r"^[\w\-\.:/]*[\d./:][\w\-\.:/]*$")


def _query_needs_llm_variants(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    if " " in q:
        return True
    return not _CODE_LIKE_RE.match(q)


# Tried capping max_tokens well below the module default (800) here,
# reasoning the output is only a small JSON object — reverted after live
# testing showed it silently breaking Arabic queries entirely (empty
# response, not truncated JSON): AGENT_MODEL (openai/gpt-oss-20b) is a
# reasoning-style model that spends part of its token budget on internal
# reasoning BEFORE the visible JSON output, and that reasoning overhead
# alone exceeded a 350-token cap for some inputs (Arabic especially),
# leaving zero budget for the actual answer. The shared get_agent_llm()
# instance (settings.LLM_MAX_TOKENS=800) is the only budget confirmed
# reliable for this model — do not lower it without re-testing Arabic
# inputs specifically, per this exact failure mode.
_query_variant_llm = get_agent_llm()


@lru_cache(maxsize=512)
def _rewrite_and_translate(query: str, lang: str, max_alternatives: int) -> Tuple[str, Tuple[str, ...], str]:
    """
    ONE combined LLM call replacing the previous two CONCURRENT calls
    (_rewrite_query + _translate, before this change) — same
    retrieval-variant value (synonym-based alternative phrasings AND a
    translation into the other language, from a single Groq round-trip
    instead of two), on AGENT_MODEL (the small/fast planner model) instead
    of GROQ_MODEL — see `_query_variant_llm`'s comment for why a lower
    max_tokens was tried and reverted here.

    The Arabic branch deliberately does NOT ask for a "corrected"
    (typo-fixed) field, unlike the English branch and unlike this
    function's pre-merge predecessor (`_rewrite_query`, which asked both
    languages for it). Live testing found Groq's OWN server-side JSON-mode
    validator reliably rejects AGENT_MODEL's response with
    `json_validate_failed` specifically for a "return the same text,
    corrected" instruction over ARABIC input — reproduced across multiple
    unrelated Arabic queries, with English inputs of the identical shape
    unaffected. Since this function swallows all exceptions (by design —
    a failed reformulation must never break retrieval), that failure was
    SILENT: it degraded to zero rewrite/translate variants for every
    Arabic query, with no visible error, no crash, and no benchmark
    check catching it (the tiny/simple test corpus still retrieved
    correctly off the untranslated original query alone) — this was only
    found by testing the raw Groq call directly outside the try/except.
    Dropping just the "corrected" field from the Arabic prompt (confirmed
    reliable across several distinct Arabic test queries afterward) keeps
    the two properties that actually matter for cross-language retrieval
    (synonym alternatives + translation — the exact case PROFILING.md's
    cross-language investigation covers) without this failure mode. The
    English branch keeps "corrected" since it was never observed to fail.

    Returns (corrected_query, alternative_phrasings, translated) —
    corrected is always "" for Arabic queries now, by design (see above).
    Never raises; falls back to ("", (), "") on any parse failure so
    retrieval still has the original query and local normalized forms to
    work with.
    """
    query = _clean(query)
    if not query:
        return "", (), ""

    if lang == "ar":
        prompt = f"""أعد نتيجة بصيغة JSON فقط (بدون أي نص خارج الـ JSON) لهذا السؤال:
- "alternatives": حتى {max_alternatives} إعادة صياغة قصيرة (جملة واحدة لكل واحدة) لنفس السؤال، باستخدام مرادفات أو مصطلحات بديلة قد تظهر في مستند يتحدث عن هذا الموضوع (مثال: "مميزات" قد تُذكر كـ"فوائد"، و"أفضل" قد تُذكر كـ"أكثر فعالية").
- "translated": ترجمة هذا السؤال إلى اللغة الإنجليزية فقط.

السؤال: {query}

أعد JSON فقط بهذا الشكل بالضبط، بدون أي شرح: {{"alternatives": ["...", "..."], "translated": "..."}}"""
    else:
        prompt = f"""Return ONLY a JSON object (no text outside the JSON) for this question:
- "corrected": the same question with only spelling mistakes fixed, meaning unchanged (or the exact same question if there are no mistakes).
- "alternatives": up to {max_alternatives} short alternative phrasings (one sentence each) of the same question, using synonyms or alternative terms that might appear in a document about this topic (e.g. "advantages" might appear as "benefits", "better" as "more effective").
- "translated": a translation of this question into Arabic only.

Question: {query}

Return ONLY JSON in exactly this shape, no explanation: {{"corrected": "...", "alternatives": ["...", "..."], "translated": "..."}}"""

    try:
        out = str(_query_variant_llm.invoke(prompt)).strip()
        match = re.search(r"\{.*\}", out, re.DOTALL)
        data = json.loads(match.group(0)) if match else {}

        corrected = _clean(str(data.get("corrected") or ""))
        if corrected.lower() == query.lower():
            corrected = ""

        alts_raw = data.get("alternatives") or []
        alternatives = tuple(
            dict.fromkeys(  # de-dupe while preserving order
                a for a in (_clean(str(a)) for a in alts_raw[:max_alternatives])
                if a and a.lower() != query.lower()
            )
        )

        translated = _clean(re.split(r"\n\n", str(data.get("translated") or ""))[0])

        return corrected, alternatives, translated
    except Exception as e:
        log.warning(f"Query rewrite+translate failed for lang={lang}: {e}")
        return "", (), ""


def _query_variants(question: str, lang: str) -> List[str]:
    """
    Build robust retrieval variants:
    - original query
    - normalized Arabic / lowercase English
    - typo-corrected query + synonym alternatives
    - a translation into whichever language the query is NOT already in
      (+ its local, non-LLM normalized/loose forms)

    ONE Groq call is made here (down from two concurrent calls, down from
    up to five before that — see Issue 1 investigation): rewriting the
    original query (typo-fix + synonym alternatives) AND translating it
    into the other language are now a single combined call — see
    `_rewrite_and_translate`'s docstring for why merging (not just running
    concurrently) is safe and preferable here, and for the move to the
    fast model. Skipped entirely for single-token identifier-like queries
    (IPs, hostnames, error codes, ...) — see `_query_needs_llm_variants`.
    Two LLM-powered passes that used to run on top of this were removed
    earlier as redundant, not just slow:
    - Translating into BOTH "en" and "ar" unconditionally used to fire
      regardless of the query's actual language, so one of the two calls
      was always given the wrong-language instruction (e.g. "translate
      this Arabic text to English" for a query that was already English)
      and produced a useless variant. Only the genuinely useful direction
      is requested now.
    - A second-pass typo-fix/synonym-expansion LLM call on top of the
      translated text was removed: translations from this model are
      reliably clean of spelling errors (they're generated, not copied
      from a typing user), and synonym coverage is already produced above
      on the original-language query — the second pass added Groq load
      without a measured retrieval-quality gain. Local normalization
      (lowercase/loose/diacritic-stripped forms) still runs on the
      translated text exactly as before.
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

    max_alts = 3 if settings.QUERY_EXPANSION_ENABLED else 1
    target_lang = "ar" if detected == "en" else "en"

    if _query_needs_llm_variants(q):
        with timing.substage("query_rewrite_and_translate"):
            fixed, alternatives, translated = _rewrite_and_translate(q, detected, max_alts)
    else:
        fixed, alternatives, translated = "", (), ""

    for alt in alternatives:
        add(alt)

    if fixed:
        add(fixed)
        add(_normalize(fixed))
        if detect_language(fixed) == "ar":
            add(_normalize_arabic(fixed))
            add(_loose_arabic(fixed))
        else:
            add(fixed.lower())
            add(_loose_english(fixed))

    if translated:
        add(translated)
        if target_lang == "ar":
            add(_normalize_arabic(translated))
            add(_loose_arabic(translated))
        else:
            add(translated.lower())
            add(_loose_english(translated))

    return variants[:22]


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


# ── Cross-encoder reranker (lazy singleton, graceful fallback) ─────────────

_cross_encoder = None
_cross_encoder_failed = False


def _get_cross_encoder():
    """
    Lazily load the shared cross-encoder reranker model (multilingual, no
    API key — same "local, free" philosophy as the embeddings model; runs
    on GPU when available, see utils/device.py, CPU otherwise). If it
    fails to load even once (offline, model not cached, etc.), permanently
    fall back to lexical-only reranking for the rest of the process
    lifetime instead of retrying every call.
    """
    global _cross_encoder, _cross_encoder_failed
    if _cross_encoder_failed or not settings.RERANK_USE_CROSS_ENCODER:
        return None
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            from utils.device import resolve_device
            device = resolve_device()
            log.info(f"Loading cross-encoder reranker '{settings.CROSS_ENCODER_MODEL}' on device={device!r} (one-time load)...")
            _cross_encoder = CrossEncoder(settings.CROSS_ENCODER_MODEL, device=device)
            log.info("Cross-encoder reranker loaded and ready.")
        except Exception as e:
            log.warning(
                f"Could not load cross-encoder '{settings.CROSS_ENCODER_MODEL}' — "
                f"falling back to lexical-only reranking: {e}"
            )
            _cross_encoder_failed = True
            return None
    return _cross_encoder


# ── Diversity selection (MMR-lite) ──────────────────────────────────────────

def _diversify(scored_docs: List[Tuple[float, Document]], top_n: int, lambda_param: float) -> List[Document]:
    """
    Greedy MMR-lite reselection over an already-scored candidate pool:
    picks `top_n` docs maximizing `lambda * relevance - (1 - lambda) *
    similarity to already-selected docs`, instead of just taking the top
    `top_n` by score. Cuts near-duplicate/overlapping chunks (e.g. adjacent
    chunks that both restate the same sentence) from crowding out distinct
    information within the same context window.

    Reuses the shared embeddings model for one batched `embed_documents`
    call over the (bounded-size) candidate pool — the same batching pattern
    already used by hybrid chunking in this module.
    """
    if len(scored_docs) <= top_n:
        return [d for _, d in scored_docs]

    texts = [_clean(d.page_content) for _, d in scored_docs]
    vectors = np.array(embeddings.embed_documents(texts))  # L2-normalized

    remaining = list(range(len(scored_docs)))
    first = max(remaining, key=lambda i: scored_docs[i][0])
    selected = [first]
    remaining.remove(first)

    while remaining and len(selected) < top_n:
        best_i, best_val = remaining[0], float("-inf")
        for i in remaining:
            relevance = scored_docs[i][0]
            similarity = max(float(np.dot(vectors[i], vectors[j])) for j in selected)
            value = lambda_param * relevance - (1 - lambda_param) * similarity
            if value > best_val:
                best_val, best_i = value, i
        selected.append(best_i)
        remaining.remove(best_i)

    return [scored_docs[i][1] for i in selected]


# ── Reranking ────────────────────────────────────────────────────────────────

def _rerank(
    question: str, variants: List[str], docs: List[Document], top_n: Optional[int] = None
) -> Tuple[List[Document], List[dict]]:
    """
    Rerank without throwing away vector-search results.
    Important for Arabic <-> English cross-language questions and typo-heavy queries.

    Blends a semantic cross-encoder score (scored against the original
    `question`, once, in a single batched call) with the existing lexical/
    bigram overlap score computed across every query variant. Falls back to
    lexical-only scoring transparently if the cross-encoder is unavailable.
    """
    top_n = top_n or settings.RERANK_TOP_N
    alpha = settings.RERANK_ALPHA

    candidates: List[Tuple[int, Document, str]] = []
    for idx, d in enumerate(docs):
        content = _clean(d.page_content)
        if _is_meaningful(content):
            candidates.append((idx, d, content))

    if not candidates:
        return [], []

    ce_scores: Optional[List[float]] = None
    cross_encoder = _get_cross_encoder()
    if cross_encoder is not None:
        try:
            with timing.substage("cross_encoder_inference"):
                pairs = [(question, content) for _, _, content in candidates]
                raw_scores = cross_encoder.predict(pairs)
                ce_scores = [float(1.0 / (1.0 + np.exp(-float(s)))) for s in raw_scores]
        except Exception as e:
            log.warning(f"Cross-encoder scoring failed, falling back to lexical-only: {e}")
            ce_scores = None
    elif settings.LOG_RETRIEVAL_DEBUG:
        log.info(
            "[retrieval-debug] cross-encoder unavailable "
            f"(RERANK_USE_CROSS_ENCODER={settings.RERANK_USE_CROSS_ENCODER}, "
            f"load_failed={_cross_encoder_failed}) -> reranking is LEXICAL-ONLY"
        )

    scored = []

    # A sheet_summary chunk (a few sample rows) can lexically/semantically
    # outrank the actual row_group chunk holding the value a question needs,
    # since both are scored the same generic way. Once at least one
    # row_group chunk is in the candidate pool, nudge summary chunks down
    # slightly so they only win on genuine relevance margin, not by
    # crowding out real row data on a near-tie.
    has_row_group = any(d.metadata.get("chunk_type") == "row_group" for _, d, _ in candidates)

    for pos, (idx, d, content) in enumerate(candidates):
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

        blended = alpha * ce_scores[pos] + (1 - alpha) * max_lex if ce_scores is not None else max_lex

        # Keep vector-search ordering as fallback tie-break.
        final_score = blended - idx * 0.001

        if has_row_group and d.metadata.get("chunk_type") == "sheet_summary":
            final_score -= 0.05

        scored.append((final_score, d, {
            "source": d.metadata.get("source", "?"),
            "page": d.metadata.get("page", 0),
            "score": round(final_score, 4),
            "preview": content[:120].replace("\n", " "),
        }))

    scored.sort(key=lambda x: x[0], reverse=True)

    if settings.RERANK_DIVERSIFY and len(scored) > top_n:
        pool_size = min(len(scored), max(top_n * 4, 12))
        shortlist = scored[:pool_size]
        debug_by_id = {id(d): dbg for _, d, dbg in shortlist}
        with timing.substage("mmr_diversification"):
            ranked = _diversify([(s, d) for s, d, _ in shortlist], top_n, settings.MMR_LAMBDA)
        debugs = [debug_by_id[id(d)] for d in ranked]
    else:
        ranked = [d for _, d, _ in scored][:top_n]
        debugs = [dbg for _, _, dbg in scored][:top_n]

    return ranked, debugs


def _add_raw_question_anchor(
    variants: List[str], planner_question: str, raw_question: Optional[str]
) -> List[str]:
    """
    Deterministically add the user's own literal wording (+ cheap LOCAL
    normalization — no LLM calls) to the retrieval variant set, alongside
    whatever `planner_question` (which may be an LLM-generated
    reformulation) already produced via `_query_variants`.

    Why: the ReAct planner is free to reformulate the retrieve query text
    (needed for legitimate coreference resolution — see prompt.py), but a
    hosted LLM's exact wording for "the same" request is not guaranteed to
    be stable run-to-run even at temperature 0 (documented already in
    Agent._correct_premature_terminal's docstring for tool CHOICE; the same
    non-determinism applies to the free-text query it writes). Since
    `_rewrite_and_translate` is cached by exact query string, so a
    different planner phrasing is a full cache miss that can pull in a
    different candidate pool for what the user experiences as "the exact
    same question". Anchoring on the raw literal text — which never comes
    from an LLM and is therefore always stable — closes that gap without
    adding any new LLM calls, extra thresholds, or reranking/MMR changes.

    No-op (returns `variants` unchanged) when `raw_question` is empty or
    already identical to `planner_question` — the common case for a
    direct, non-paraphrased question, where nothing needs anchoring.
    """
    raw_question = _clean(raw_question)
    if not raw_question or raw_question.lower() == _clean(planner_question).lower():
        return variants

    raw_lang = detect_language(raw_question)
    extra = [raw_question, _normalize(raw_question)]
    extra.append(_normalize_arabic(raw_question) if raw_lang == "ar" else raw_question.lower())

    out = list(variants)
    for v in extra:
        v = _clean(v)
        if v and v not in out:
            out.append(v)
    return out


def _retrieve(
    question: str,
    lang: str,
    conversation_id: str,
    k: Optional[int] = None,
    top_n: Optional[int] = None,
    source_filter: Optional[str] = None,
    raw_question: Optional[str] = None,
) -> Tuple[List[Document], str, List[dict]]:
    """
    Retrieve + rerank documents for a question.

    `k` overrides how many candidates are pulled per query variant
    (defaults to settings.RETRIEVER_K); `top_n` overrides how many
    reranked documents are returned (defaults to settings.RERANK_TOP_N).
    `source_filter`, if given, restricts results to chunks whose
    metadata["source"] exactly matches it (used by topic-scoped report
    generation to search within one named document instead of the whole
    knowledge base) — applied client-side, AFTER the vector search.
    `conversation_id` is kept as a parameter (existing callers all pass
    it, and it's still used elsewhere — see `delete_conversation_documents`,
    `reindex_document`) but is NOT used to scope this search: every
    uploaded document is retrievable by every conversation, regardless of
    which session uploaded it. (Per-conversation retrieval isolation was
    removed — a document a staff member uploads needs to stay findable
    after their browser tab closes, not just within the session that
    uploaded it.) Used internally by `ask_question` and exposed via
    `retrieve()` for the agent's retrieve tool. `raw_question` — see
    `_add_raw_question_anchor`.

    Returns (ranked_docs, debug_str, per_doc_debug) — `per_doc_debug` is
    the same length/order as `ranked_docs` and carries each doc's
    "score" (0-1), which callers that need to filter by relevance (e.g.
    topic-scoped report generation, which asks for many more candidates
    than a chat answer needs and must not silently pad itself out with
    barely-relevant chunks) can use instead of blindly taking the top_n.
    """
    vdb = _get_vector_db()
    search_k = k or settings.RETRIEVER_K
    # No conversation_id filter here (see docstring above) — search spans
    # every indexed document, not just this conversation's own uploads.
    conv_filter = None

    with timing.stage("query_variant_generation_total"):
        variants = _query_variants(question, lang)
        variants = _add_raw_question_anchor(variants, question, raw_question)

    # _query_variants already dedupes exact-string repeats, but e.g. "Query"
    # and a later ".lower()" variant "query" can both survive as distinct
    # entries without being meaningfully different vector searches — collapse
    # those before spending a Qdrant round-trip on each one.
    seen_ci = set()
    deduped_variants = []
    for v in variants:
        key = v.lower()
        if key not in seen_ci:
            seen_ci.add(key)
            deduped_variants.append(v)
    variants = deduped_variants

    if settings.LOG_RETRIEVAL_DEBUG:
        log.info(f"[retrieval-debug] original_query={question!r} detected_lang={lang!r}")
        log.info(f"[retrieval-debug] query_variants ({len(variants)}): {variants}")

    def _search_by_vector(q: str, vector: List[float]) -> List[Document]:
        # Takes a PRECOMPUTED vector (see the batched embed_queries() call
        # below) instead of embedding `q` itself here. Searching Qdrant
        # with an already-computed vector is cheap I/O-bound work, safe to
        # fan out across threads; embedding text is the expensive part and
        # is done once, batched, up front (see comment at the call site).
        try:
            hits = vdb.similarity_search_with_score_by_vector(
                vector, k=search_k, filter=conv_filter
            )
        except Exception as e:
            log.error(f"Retrieval error for '{q}': {e}")
            return []
        docs = []
        for doc, score in hits:
            doc.metadata["_vector_score"] = float(score)
            doc.metadata["_matched_variant"] = q
            docs.append(doc)
        if settings.LOG_RETRIEVAL_DEBUG:
            preview = ", ".join(
                f"{os.path.basename(d.metadata.get('source', '?'))}"
                f"(p{d.metadata.get('page', 0)}, score={d.metadata.get('_vector_score', 0):.4f})"
                for d in docs
            ) or "(none)"
            log.info(f"[retrieval-debug] variant={q!r} -> {len(docs)} hits: {preview}")
        return docs

    all_docs: List[Document] = []
    with timing.stage("embedding_and_qdrant_retrieval"):
        # Embed every query variant in ONE batched call instead of one
        # embed call per variant (even concurrently): many small
        # single-item embedding calls fired at once is fine on CPU
        # (independent cores) but is much slower on GPU than one batched
        # call of the same total size, since a single device serializes
        # concurrently-submitted work rather than truly parallelizing it
        # (measured ~10x slower — see PROFILING.md). This is the expensive
        # step, done once, up front.
        #
        # embedding_ms / qdrant_search_ms below are ADDITIONAL, finer-
        # grained substages nested inside the existing
        # embedding_and_qdrant_retrieval stage (unchanged) — each
        # substage() call here is made once by THIS single calling thread
        # (embedding is one batched call; the Qdrant fan-out is measured
        # as one wall-clock block around the whole _run_concurrent call,
        # not per-worker-thread), so this adds no new concurrent-write
        # risk. Pure instrumentation: same calls, same arguments, same
        # batching/concurrency as before.
        with timing.substage("embedding_ms"):
            vectors = embeddings.embed_queries(variants)

        # Only the Qdrant lookups (cheap, I/O-bound) are fanned out across
        # threads now — no embedding work left inside them to contend over.
        with timing.substage("qdrant_search_ms"):
            for docs in _run_concurrent(
                [lambda q=q, v=v: _search_by_vector(q, v) for q, v in zip(variants, vectors)]
            ):
                all_docs.extend(docs)

    # Fallback for very typo-heavy or short queries
    if not all_docs:
        simplified = " ".join(
            re.findall(r"[\u0600-\u06FFa-zA-Z]{2,}", _normalize(question))[:8]
        )
        if simplified:
            all_docs.extend(_search_by_vector(simplified, embeddings.embed_query(simplified)))

    all_docs = _deduplicate_retrieved(all_docs)

    if source_filter:
        all_docs = [d for d in all_docs if d.metadata.get("source") == source_filter]

    if not all_docs:
        return [], "no docs retrieved", []

    # Spreadsheet questions often need several row_group chunks to see all
    # the relevant rows — the default top_n (tuned for prose chunks) would
    # otherwise silently drop most of a sheet's matching rows. Widen the
    # final result count whenever Excel-sourced chunks made it into the
    # candidate pool at all, regardless of what top_n the caller requested.
    effective_top_n = top_n or settings.RERANK_TOP_N
    if any(d.metadata.get("file_type") == "excel" for d in all_docs):
        effective_top_n = max(effective_top_n, settings.EXCEL_RERANK_TOP_N)

    if settings.LOG_RETRIEVAL_DEBUG:
        before = sorted(
            (
                (d.metadata.get("_vector_score", 0.0), os.path.basename(d.metadata.get("source", "?")), d.metadata.get("page", 0))
                for d in all_docs
            ),
            reverse=True,
        )[:10]
        log.info(f"[retrieval-debug] top vector-similarity scores BEFORE rerank: {before}")

    with timing.stage("reranking"):
        ranked, debugs = _rerank(question, variants, all_docs, top_n=effective_top_n)

    if settings.LOG_RETRIEVAL_DEBUG:
        after = [(d["score"], d["source"], d["page"]) for d in debugs[:10]]
        log.info(f"[retrieval-debug] top blended scores AFTER rerank: {after}")

    if not debugs:
        return [], "no docs retrieted after rerank", []

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
        if settings.LOG_RETRIEVAL_DEBUG:
            log.info(
                f"[retrieval-debug] REJECTED: top_score={top_score} < "
                f"CONFIDENCE_THRESHOLD={settings.CONFIDENCE_THRESHOLD} -> no context sent to LLM"
            )
        return [], f"{debug_str}\n[top score {top_score} below CONFIDENCE_THRESHOLD={settings.CONFIDENCE_THRESHOLD} — treating as no relevant match]", []

    debug_str = "\n".join(
        f"Rank {i+1}: {d['source']} p{d['page']} score={d['score']} | {d['preview'][:80]}"
        for i, d in enumerate(debugs)
    )

    return ranked, debug_str, debugs


# ── Prompt & cleanup ────────────────────────────────────────────────────────────

def build_prompt(context: str, question: str, lang: str) -> str:
    # Retrieved chunk text originates from user-uploaded documents, not
    # from the developer or the system — wrap it with explicit
    # untrusted-data framing so an injected instruction inside a document
    # ("ignore previous instructions...") is treated as content to report
    # on, never as something to obey. See utils/prompt_safety.py.
    context = wrap_untrusted_context(context, lang)

    # Compressed as part of a latency/token-optimization pass — see the
    # speed-optimization report. Only STYLE/formatting rules were merged
    # (old 1+2+3 -> 1, old 9+9b+9c -> 5, old 11+12+13 -> 7); the grounding
    # / anti-hallucination / refusal rules (old 4/5/6, now 2/3) are kept
    # essentially verbatim — these are the actual quality gate (grounded
    # vs. ungrounded refusal correctness) and the highest-risk rules to
    # weaken. Re-verify refusal behavior after any further edit here — see
    # scripts/evaluate_agent_token_and_quality.py's ungrounded scenarios.
    if lang == "ar":
        return f"""أنت نظام استخراج معلومات متقدم. مهمتك: استخرج إجابة كاملة ومفصّلة من السياق المقدم فقط.

**قواعد الإجابة — يجب اتباعها بدقة:**

1. اقرأ السياق كاملاً قبل الكتابة. قد يكون السؤال بالعربية والسياق بالإنجليزية أو العكس، وقد يحتوي السؤال على أخطاء إملائية أو حروف ناقصة أو كلمات عامية — افهم المعنى المقصود من السياق رغم ذلك.
2. أجب من السياق فقط إذا كان يحتوي بشكل مباشر ومحدد على المعلومة المطلوبة في السؤال — وليس مجرد موضوع مشابه أو قريب.
3. إذا كان السياق لا يحتوي على المعلومة المحددة المطلوبة، يجب أن تقول "المعلومة غير موجودة في الملفات المرفوعة." لا تجب إجابة جزئية بالاعتماد على فقرة قريبة الموضوع فقط، ولا تسد الفجوة بمعرفتك الخاصة — حتى لو كنت متأكدًا أنها معلومة صحيحة في الواقع. لا تستخدم أي معرفة خارجية خارج السياق تحت أي ظرف، حتى للأسئلة التي تستطيع الإجابة عنها بسهولة بنفسك. إذا لم تكن المعلومة في السياق، فلا تضعها في الإجابة.
4. لا تبدأ بعبارات مثل "بناءً على السياق" أو "وفقاً للمعلومات"، ولا تكرر السؤال في الإجابة.
5. اذكر الأرقام والتواريخ والأسماء والمعادلات والصيغ والوحدات والمصطلحات التقنية كما وردت بالضبط في السياق؛ لا تعيد صياغتها أو تقرّبها أو تبسّطها، ولا تخترع رقمًا أو اسمًا أو مصطلحًا غير موجود، حتى لو كان ذلك لسد فجوة أو لجعل الإجابة تبدو مكتملة.
6. أجب دائمًا بنفس لغة سؤال المستخدم في هذه الرسالة، بغض النظر عن لغة المستند المصدر. لا تُغيّر اللغة من تلقاء نفسك. حافظ على المصطلحات الإنجليزية المهمة كما هي عند الحاجة.
7. أجب مباشرة وبإيجاز، وطابق حجم الإجابة مع حجم السؤال: سؤال بسيط ومباشر يستحق إجابة قصيرة بدون إطالة؛ سؤال تقني أو مفاهيمي معقد يستحق شرحًا واضحًا بالقدر اللازم فقط. أضف مثالًا فقط إذا كان المفهوم غير بديهي أو طلبه المستخدم صراحة. اكتب نثرًا طبيعيًا منظمًا، لا تنسيقًا ثابتًا مثل "الشرح:/المثال:" — واستخدم نقاطًا فقط عندما يكون المحتوى قائمة فعلية.
8. بعض أجزاء السياق مصدرها ملف إكسل/جدول بيانات، ويظهر ذلك في ترويسة الجزء (مثل "Sheet: Sales | Rows 12-20") وفي نص الصف بصيغة "Row N: العمود: القيمة | العمود: القيمة". اقرأ كل صف كسجل بيانات منفصل تربط فيه كل قيمة باسم عمودها. إذا احتجت لتجميع قيم من عدة صفوف، استخدم فقط الصفوف الظاهرة فعليًا في السياق، ووضّح أن إجابتك مبنية على الصفوف المتاحة فقط وليست شاملة لكل البيانات.

**السياق:**
{context}

**السؤال:**
{question}

**الإجابة:**"""

    return f"""You are an advanced information extraction system. Your task is to extract a complete, well-structured answer from the provided context only.

**Answering rules — follow precisely:**

1. Read the entire context before writing. The question and context may be in different languages (Arabic/English), and the question may contain spelling mistakes, missing letters, dialect words, or mixed terms — infer the intended meaning from the context regardless.
2. Answer using the context ONLY if it specifically and directly contains the information the question is asking for — not merely a related or similar topic.
3. If the context does not contain the specific information requested, you MUST say "The information is not available in the uploaded files." Do not partially answer using a superficially related passage, and do not fill the gap with your own knowledge — even if you are confident it is correct real-world information. This system must answer strictly from the uploaded documents, nothing else. Do not use external knowledge outside the context under any circumstance, even for questions you could easily answer yourself (general facts, definitions, current events, etc.). If it isn't in the context, it isn't in the answer.
4. Do not open with "Based on the context" or "According to the information", and do not repeat the question.
5. State numbers, dates, names, equations, formulas, units, and technical terms exactly as they appear in the context — do not paraphrase, round, or simplify them, and never invent one that isn't present, even to fill a gap or sound complete.
6. Always answer in the same language as the user's question in this message, regardless of the source document's language. Never switch language on your own. Preserve important Arabic or English technical terms when needed.
7. Answer directly and concisely, matching length to the question: a simple, direct question (a definition, a date, a name, a number) deserves a short, direct answer with no padding; a complex or technical/conceptual question deserves a clear explanation, only as long as it needs to be. Include an example only when the concept is genuinely non-obvious and it would meaningfully aid understanding, or the user explicitly asked for one. Write natural, well-organized prose, not a forced "Explanation:/Example:" structure — use bullet points only when the content is genuinely a list.
8. Some context chunks come from an Excel/spreadsheet file — you can tell from the chunk header (e.g. "Sheet: Sales | Rows 12-20") and from row text formatted as "Row N: Column: Value | Column: Value". Read each row as its own record, matching every value to its column name. If the question requires aggregating across several rows (a total, a count, a maximum, etc.), use only the rows actually present in the context; if the context clearly only contains some of a sheet's rows rather than the whole table, say your answer is based only on the rows available rather than implying it covers the entire dataset.

**Context:**
{context}

**Question:**
{question}

**Answer:**"""


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


def _save_uploaded_file(filename: str, data: bytes, fhash: str) -> Optional[str]:
    """
    Upload the original file bytes to the MinIO uploads bucket instead of
    local disk. Returns the MinIO object key (stored in the registry as
    "stored_path"), or None if object storage is unavailable (the `minio`
    package isn't installed, or the server isn't reachable) — ingestion
    (chunking/embedding/retrieval) still proceeds without it; only
    downloading the original file and PDF report generation need it.
    """
    safe_name = _safe_filename(filename)
    stem, ext = os.path.splitext(safe_name)

    object_name = f"{stem}_{fhash[:10]}{ext}"
    content_type = get_content_type(filename)

    try:
        storage_service.upload_bytes(
            settings.MINIO_BUCKET_UPLOADS, object_name, data, content_type=content_type
        )
        return object_name
    except Exception as e:
        log.warning(
            f"Object storage unavailable — '{filename}' will be indexed and "
            f"searchable, but its original file won't be downloadable: {e}"
        )
        return None


def get_stored_file_bytes(object_name: str) -> bytes:
    """Download a previously uploaded file's raw bytes from MinIO."""
    return storage_service.download_bytes(settings.MINIO_BUCKET_UPLOADS, object_name)


def list_stored_files(conversation_id: Optional[str] = None) -> list[dict]:
    """
    List registry entries, optionally restricted to one conversation_id's
    own uploads. `conversation_id=None` (the default, used by the existing
    /stored-files route and the report tool's whole-document resolution)
    lists every uploaded file across every conversation — that surface
    remains intentionally out of scope for this change (see the Document
    Isolation implementation notes); only vector retrieval is scoped.
    """
    registry = _load_registry()
    files = []

    for _, info in registry.items():
        if conversation_id is not None and info.get("conversation_id") != conversation_id:
            continue
        object_name = info.get("stored_path")
        files.append({
            "filename": info.get("filename"),
            "stored_path": object_name,
            "file_type": info.get("file_type"),
            "chunks": info.get("chunks", 0),
            "processed_at": info.get("processed_at"),
            "conversation_id": info.get("conversation_id"),
            "document_id": info.get("document_id"),
            "download_url": storage_service.presigned_url(
                settings.MINIO_BUCKET_UPLOADS, object_name
            ) if object_name else None,
        })

    files.sort(key=lambda x: x.get("processed_at") or "", reverse=True)
    return files


def delete_conversation_documents(conversation_id: str) -> int:
    """
    Physically delete every indexed chunk belonging to `conversation_id`
    from Qdrant, and drop that conversation's entries from the file
    registry — called on conversation reset (see routes/chat.py::chat_reset)
    so a reset conversation's previously uploaded documents become
    genuinely inaccessible to future retrievals under that same
    conversation_id, not merely hidden.

    Physical deletion IS feasible here (Qdrant's client supports deleting
    points by the exact same metadata filter used at query time, even in
    the local/embedded mode this app uses — verified directly against the
    installed qdrant-client), so there is no "soft delete"/orphaned-vector
    fallback needed: this is scoped by the identical
    `metadata.conversation_id` filter `_retrieve()` uses, so it is
    impossible for this call to remove another conversation_id's points.

    Registry entries for OTHER conversation_ids that happen to reference
    the same underlying MinIO object (byte-identical file uploaded by more
    than one conversation) are left untouched — only entries whose own
    `conversation_id` matches are dropped, and the MinIO original itself
    is not deleted (out of scope — this only affects retrieval-eligible
    state, not raw file storage/downloads).

    Safe to call for a conversation with no indexed documents (no-op).
    Returns the number of registry entries removed.
    """
    if is_ready():
        conv_filter = _conversation_filter(conversation_id)
        try:
            with_retries(
                lambda: get_client().delete(
                    collection_name=get_collection_name(), points_selector=conv_filter
                ),
                what=f"delete vectors for conversation_id={conversation_id!r}",
            )
        except Exception as e:
            log.error(f"Failed to delete vectors for conversation_id={conversation_id!r}: {e}")
            raise

    registry = _load_registry()
    keys_to_remove = [
        k for k, info in registry.items() if info.get("conversation_id") == conversation_id
    ]
    for k in keys_to_remove:
        del registry[k]
    if keys_to_remove:
        _save_registry(registry)
        log.info(
            f"Document isolation: removed {len(keys_to_remove)} registry entr(y/ies) "
            f"for conversation_id={conversation_id!r} on reset."
        )

    return len(keys_to_remove)


def delete_document(document_id: str, conversation_id: str) -> int:
    """
    Task 4 — physically delete every indexed chunk belonging to exactly
    one document_id from Qdrant, plus its registry entry, WITHOUT
    affecting any other document (including other documents in the same
    conversation) or any other conversation's documents.

    Scoped by `_document_filter(document_id, conversation_id)` — the same
    (document_id AND conversation_id) filter `reindex_document` uses — so
    a caller can never delete a document belonging to a different
    conversation even if it somehow knew that document_id.

    Idempotent: deleting a document_id that matches zero Qdrant points
    (already deleted, or never existed) is a normal Qdrant no-op, not an
    error — repeated calls are safe and simply return 0 on every call
    after the first.

    Returns the number of Qdrant points (chunks) actually removed — 0 on
    a second/repeated call, not merely whether a registry entry existed
    (the registry and Qdrant could in principle disagree; the vector
    count is the ground truth for "is this document still retrievable").
    """
    removed = 0
    if is_ready():
        doc_filter = _document_filter(document_id, conversation_id)
        try:
            removed = with_retries(
                lambda: get_client().count(
                    collection_name=get_collection_name(), count_filter=doc_filter, exact=True
                ).count,
                what=f"count vectors for document_id={document_id!r} before delete",
            )
            with_retries(
                lambda: get_client().delete(
                    collection_name=get_collection_name(), points_selector=doc_filter
                ),
                what=f"delete vectors for document_id={document_id!r}",
            )
        except Exception as e:
            log.error(f"Failed to delete vectors for document_id={document_id!r}: {e}")
            raise

    found = _find_registry_key_and_entry_by_document_id(document_id, conversation_id)
    if found is not None:
        key, _entry = found
        registry = _load_registry()
        if key in registry:
            del registry[key]
            _save_registry(registry)

    log.info(
        f"Deleted document_id={document_id!r} (conversation_id={conversation_id!r}) — "
        f"{removed} chunk(s) removed."
    )

    return removed


def find_registry_entry(filename: str) -> Optional[dict]:
    """Look up a stored file's registry entry (incl. its MinIO object key)
    by original filename. Used by the report-generation endpoint."""
    registry = _load_registry()
    for info in registry.values():
        if info.get("filename") == filename:
            return info
    return None


def _find_registry_key_and_entry_by_document_id(
    document_id: str, conversation_id: str
) -> Optional[Tuple[str, dict]]:
    """
    Look up a registry (key, entry) pair by document_id, scoped to
    conversation_id (the same safety scoping `_document_filter` applies to
    the Qdrant side) — used by `reindex_document` and `delete_document` so
    neither can act on a document_id belonging to a different conversation.
    """
    registry = _load_registry()
    for key, info in registry.items():
        if info.get("document_id") == document_id and info.get("conversation_id") == conversation_id:
            return key, info
    return None


def find_registry_entry_by_document_id(document_id: str, conversation_id: str) -> Optional[dict]:
    """Public counterpart of `_find_registry_key_and_entry_by_document_id`
    that returns just the entry — used by routes to check existence/build
    a response without needing the internal registry key."""
    found = _find_registry_key_and_entry_by_document_id(document_id, conversation_id)
    return found[1] if found else None


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
    if not entry.get("stored_path"):
        raise FileNotFoundError(
            f"'{filename}' has no stored original file — object storage was "
            f"unavailable when it was uploaded, so a report can't be generated for it."
        )

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


# ── Document loading from bytes ────────────────────────────────────────────────
# Per-extension parsing lives in loaders/ (pdf_loader, docx_loader, text_loader,
# image_loader, excel_loader) — this function just dispatches via
# loaders.registry and applies the bilingual enrichment pass uniformly to
# whatever any loader returns.

def _load_document_from_bytes(filename: str, data: bytes) -> List[Document]:
    """Load (via the loaders registry) and enrich a document from its raw bytes."""
    try:
        raw_docs = _dispatch_load(filename, data)
    except Exception as e:
        log.error(f"load_document error ({filename}): {e}")
        raw_docs = []

    enriched = []
    for d in raw_docs:
        content = _enrich(d.page_content)
        if content:
            d.page_content = content
            enriched.append(d)

    log.info(f"Loaded '{filename}' -> {len(enriched)} docs")
    return enriched


# ── Semantic chunking (opt-in, see settings.CHUNKING_STRATEGY) ──────────────

# Splits on Arabic/English sentence terminators while keeping the
# terminator attached to the sentence it ends.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?؟۔])\s+|\n{2,}")


def _split_into_sentences(text: str) -> List[str]:
    parts = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text)]
    return [p for p in parts if p]


def _semantic_split_documents(docs: List[Document]) -> List[Document]:
    """
    Groups each document's sentences into chunks based on embedding-similarity
    breakpoints (adjacent sentence windows that shift topic get split), instead
    of fixed character counts.

    Performance note: this does ONE extra batched `embed_documents` call per
    source document (over sentence windows, not raw sentences — controlled by
    SEMANTIC_CHUNK_BUFFER_SIZE — to keep the number of vectors small), reusing
    the already-loaded local embedding model. Any chunk that still ends up
    larger than SEMANTIC_CHUNK_MAX_CHARS is handed to the same
    RecursiveCharacterTextSplitter used by the default strategy, so output
    size stays bounded either way.
    """
    buffer_size = max(1, settings.SEMANTIC_CHUNK_BUFFER_SIZE)
    percentile = settings.SEMANTIC_CHUNK_BREAKPOINT_PERCENTILE
    max_chars = settings.SEMANTIC_CHUNK_MAX_CHARS

    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n\n", "\n\n", "\n", ".", " ", ""],
    )

    out: List[Document] = []

    for doc in docs:
        sentences = _split_into_sentences(doc.page_content)

        if len(sentences) <= 1:
            if doc.page_content.strip():
                out.append(doc)
            continue

        # Combine neighboring sentences into overlapping windows so each
        # embedding captures local context, not an isolated one-liner.
        windows = [
            " ".join(sentences[max(0, i - buffer_size): i + buffer_size + 1])
            for i in range(len(sentences))
        ]

        vectors = np.array(embeddings.embed_documents(windows))

        # Vectors are L2-normalized (see embeddings_provider.py), so cosine
        # distance between consecutive windows is just 1 - dot product.
        sims = np.sum(vectors[:-1] * vectors[1:], axis=1)
        distances = 1.0 - sims

        if len(distances) == 0:
            out.append(doc)
            continue

        threshold = np.percentile(distances, percentile)
        breakpoints = set(int(i) for i in np.where(distances > threshold)[0])

        group: List[str] = []
        for i, sent in enumerate(sentences):
            group.append(sent)
            if i in breakpoints or i == len(sentences) - 1:
                chunk_text = " ".join(group).strip()
                group = []
                if not chunk_text:
                    continue

                # Bug fix: this loop used to compute chunk_text and then
                # discard it — no Document was ever appended to `out`, so
                # every multi-sentence document silently produced zero
                # chunks under CHUNKING_STRATEGY=semantic (single-sentence
                # documents "worked" only because they hit the early
                # `out.append(doc)` return above, never reaching this loop).
                # Each chunk gets its OWN metadata dict copy — sharing
                # doc.metadata across chunks would let a later per-chunk
                # mutation (e.g. chunk_index in update_db_files) bleed
                # into every other chunk from the same source document.
                if len(chunk_text) > max_chars:
                    out.extend(
                        fallback_splitter.split_documents(
                            [Document(page_content=chunk_text, metadata=dict(doc.metadata))]
                        )
                    )
                else:
                    out.append(Document(page_content=chunk_text, metadata=dict(doc.metadata)))

    return out


def _hybrid_split_documents(docs: List[Document]) -> List[Document]:
    """
    Fast semantic-aware chunking: recursive (character-based, no embedding)
    splitting into small base chunks, then ONE batched embedding call over
    those base chunks — merging adjacent chunks (from the same source file)
    whose embeddings are similar into a single chunk.

    Why this is much faster than `_semantic_split_documents`: the number of
    embedding calls scales with the number of ~HYBRID_BASE_CHUNK_SIZE-sized
    base chunks, not with the number of sentences — typically 4-6x fewer
    vectors for the same document, since a base chunk already holds several
    sentences. It's also more robust on messy PDF/slide text where sentence
    punctuation is inconsistent (recursive splitting doesn't depend on it).
    """
    base_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.HYBRID_BASE_CHUNK_SIZE,
        chunk_overlap=0,
        separators=["\n\n\n", "\n\n", "\n", ".", " ", ""],
    )

    base_chunks = [
        c for c in base_splitter.split_documents(docs) if _is_meaningful(c.page_content)
    ]

    if len(base_chunks) <= 1:
        return base_chunks

    vectors = np.array(embeddings.embed_documents([c.page_content for c in base_chunks]))

    threshold = settings.HYBRID_MERGE_SIMILARITY_THRESHOLD
    max_chars = settings.HYBRID_CHUNK_MAX_CHARS

    merged: List[Document] = []
    cur_texts = [base_chunks[0].page_content]
    cur_meta = dict(base_chunks[0].metadata)

    for i in range(1, len(base_chunks)):
        same_source = base_chunks[i].metadata.get("source") == cur_meta.get("source")
        # Vectors are L2-normalized, so dot product == cosine similarity.
        sim = float(np.dot(vectors[i - 1], vectors[i])) if same_source else -1.0
        candidate_len = sum(len(t) for t in cur_texts) + 1 + len(base_chunks[i].page_content)

        if same_source and sim >= threshold and candidate_len <= max_chars:
            cur_texts.append(base_chunks[i].page_content)
        else:
            merged.append(Document(page_content=" ".join(cur_texts).strip(), metadata=cur_meta))
            cur_texts = [base_chunks[i].page_content]
            cur_meta = dict(base_chunks[i].metadata)

    if cur_texts:
        merged.append(Document(page_content=" ".join(cur_texts).strip(), metadata=cur_meta))

    return merged


def _chunk_documents(all_docs: List[Document]) -> List[Document]:
    """
    Split loaded documents into final, embeddable chunks per
    settings.CHUNKING_STRATEGY. Shared by both fresh ingestion
    (update_db_files) and re-indexing (reindex_document) so the two paths
    can never drift into different chunking behavior for the same
    configuration — extracted from what used to be inline-only logic in
    update_db_files.

    Excel-sourced documents bypass chunking entirely — loaders/excel_loader.py
    already returns final, deliberately-sized sheet_summary/row_group
    chunks; re-splitting them would cut row boundaries and discard their
    chunk_type/row_range metadata.
    """
    excel_docs = [d for d in all_docs if d.metadata.get("file_type") == "excel"]
    other_docs = [d for d in all_docs if d.metadata.get("file_type") != "excel"]

    raw_chunks: List[Document] = []
    strategy = (settings.CHUNKING_STRATEGY or "recursive").lower()

    def _recursive_split(docs: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n\n", "\n\n", "\n", ".", " ", ""],
        )
        return splitter.split_documents(docs)

    if other_docs:
        if strategy == "semantic":
            try:
                raw_chunks = _semantic_split_documents(other_docs)
            except Exception:
                log.exception("Semantic chunking failed — falling back to recursive splitter.")
                strategy = "recursive"
            else:
                if not raw_chunks:
                    # A non-empty input that yields zero chunks must never
                    # silently pass as a successful, zero-content index —
                    # fall back to the recursive splitter instead of letting
                    # this document vanish (see the semantic-chunking
                    # append-bug fix in _semantic_split_documents).
                    log.error(
                        "Semantic chunking produced 0 chunks from %d non-empty "
                        "document(s) — falling back to recursive splitter.",
                        len(other_docs),
                    )
                    strategy = "recursive"
        elif strategy == "hybrid":
            try:
                raw_chunks = _hybrid_split_documents(other_docs)
            except Exception:
                log.exception("Hybrid chunking failed — falling back to recursive splitter.")
                strategy = "recursive"
            else:
                if not raw_chunks:
                    log.error(
                        "Hybrid chunking produced 0 chunks from %d non-empty "
                        "document(s) — falling back to recursive splitter.",
                        len(other_docs),
                    )
                    strategy = "recursive"

        if strategy not in ("semantic", "hybrid"):
            raw_chunks = _recursive_split(other_docs)

    chunks = [c for c in raw_chunks if _is_meaningful(c.page_content)]
    chunks += [d for d in excel_docs if _is_meaningful(d.page_content)]
    return chunks


# ── Public API: ingestion ───────────────────────────────────────────────────────

def _registry_key(conversation_id: str, file_hash: str) -> str:
    """
    Registry dedup key. Deliberately scoped by (conversation_id, hash),
    NOT hash alone: a global hash-only key would mean the first
    conversation to ever upload a given file's exact bytes "claims" it,
    and every other conversation uploading the identical file afterward
    would be silently skipped as "already processed" — meaning that
    second conversation would never get its OWN indexed, conversation_id-
    tagged chunks in Qdrant at all, breaking retrieval for it entirely.
    Scoping by conversation_id means each conversation independently
    re-embeds and indexes a file it uploads, even if another conversation
    already uploaded byte-identical content — see Document Isolation.
    """
    return f"{conversation_id}:{file_hash}"


def update_db_files(
    files: List[dict[str, Any]],
    conversation_id: str,
    on_progress: Optional[Callable[[str], None]] = None,
) -> int:
    """
    Ingest a list of {'filename': str, 'data': bytes} dicts into Qdrant,
    tagging every resulting chunk with `conversation_id` (and a fresh
    per-upload `document_id`) so retrieval can later be scoped to exactly
    this conversation's own documents — see Document Isolation.
    Also uploads the original file bytes to MinIO (settings.MINIO_BUCKET_UPLOADS).
    Returns total number of chunks added.

    `on_progress`, if given, is called once per stage ("parsing", "chunking",
    "embedding") — used by the background upload job (routes/upload.py) to
    report real progress instead of a single opaque "processing" spinner.
    Failures in the callback itself are logged and swallowed; they must
    never abort ingestion.
    """
    def _report(stage: str) -> None:
        if on_progress is None:
            return
        try:
            on_progress(stage)
        except Exception:
            log.warning(f"on_progress callback failed for stage '{stage}'", exc_info=True)

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

        if _registry_key(conversation_id, h) in registry:
            skipped.append(filename)
            log.info(f"Skip already processed for conversation_id={conversation_id!r}: {filename}")
        else:
            new_files.append((filename, data, h))

    if not new_files:
        log.info(f"No new files — skipped: {skipped}")
        return 0

    _report("parsing")

    all_docs: List[Document] = []
    per_file_info: dict = {}

    for filename, data, fhash in new_files:
        stored_path = _save_uploaded_file(filename, data, fhash)
        document_id = str(uuid.uuid4())

        docs = _load_document_from_bytes(filename, data)

        for d in docs:
            d.metadata["stored_path"] = stored_path
            # Document-isolation metadata — required on every chunk so
            # retrieval's Qdrant filter (see _conversation_filter) can
            # restrict a search to exactly this conversation's documents.
            d.metadata["conversation_id"] = conversation_id
            d.metadata["document_id"] = document_id

        all_docs.extend(docs)

        per_file_info[filename] = {
            "hash": fhash,
            "stored_path": stored_path,
            "docs_count": len(docs),
            "document_id": document_id,
        }

    all_docs = _deduplicate(all_docs)

    if not all_docs:
        log.warning("No valid documents extracted from uploaded files.")
        return 0

    _report("chunking")

    chunks = _chunk_documents(all_docs)

    if not chunks:
        log.warning("No meaningful chunks generated.")
        return 0

    _report("embedding")

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

        registry[_registry_key(conversation_id, fhash)] = {
            "filename": filename,
            "stored_path": info["stored_path"],
            "file_type": _get_file_type(filename),
            "chunks": cps.get(filename, 0),
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "conversation_id": conversation_id,
            "document_id": info["document_id"],
        }

    _save_registry(registry)

    log.info(
        f"Added {len(chunks)} chunks from {len(new_files)} new file(s). "
        f"Skipped: {skipped}"
    )

    _report("done")

    return len(chunks)


def reindex_document(
    document_id: str,
    conversation_id: str,
    new_filename: Optional[str] = None,
    new_data: Optional[bytes] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> int:
    """
    Task 3 — safely re-index an already-ingested document: reprocess it
    (either its original stored bytes, or replacement `new_data`) through
    the exact same chunk -> embed pipeline a fresh upload uses
    (`_chunk_documents`, shared with `update_db_files`), then swap it into
    Qdrant so the OLD version's chunks can never coexist with the NEW
    version's, and the document is never left with zero retrievable
    content either.

    Ordering is deliberately "prepare & validate new -> remove old ->
    index new" rather than the naive "remove old -> reprocess -> index
    new": the new chunks are computed and confirmed non-empty BEFORE the
    old Qdrant points are deleted. This still guarantees the required
    invariant (old and new are never BOTH retrievable for any meaningful
    window — the delete and the insert happen back-to-back with nothing
    else in between) while additionally guaranteeing reprocessing can
    never wipe a document's index down to nothing if it fails partway.

    `document_id` is preserved across the reindex (NOT regenerated) — it
    is the stable identifier callers (and any future production
    integration) use to refer to "this document" across its whole
    lifecycle, independent of how many times its content/config changes.

    Covers every reindex trigger in the task spec: chunking config
    changed, embedding model changed, OCR output changed, a processing
    bug was fixed (`new_data=None` — re-downloads and reprocesses the
    SAME original bytes through whatever the CURRENT pipeline now does),
    or the document's content itself was replaced (`new_data` provided).

    Raises ValueError if no registry entry exists for document_id under
    conversation_id (including a document_id that belongs to a DIFFERENT
    conversation_id — deliberately indistinguishable from "not found" so
    this can't be used to probe another conversation's document ids), or
    if no source bytes are available at all (neither new_data nor a
    previously-stored original — e.g. object storage was unavailable at
    the original upload). Raises RuntimeError if reprocessing yields zero
    extracted content or zero chunks — a reindex must never silently
    leave a document with no retrievable content.

    Returns the new chunk count.
    """
    def _report(stage: str) -> None:
        if on_progress is None:
            return
        try:
            on_progress(stage)
        except Exception:
            log.warning(f"on_progress callback failed for stage '{stage}'", exc_info=True)

    found = _find_registry_key_and_entry_by_document_id(document_id, conversation_id)
    if found is None:
        raise ValueError(
            f"No document found with document_id={document_id!r} for "
            f"conversation_id={conversation_id!r}."
        )
    old_key, entry = found

    _report("parsing")

    filename = new_filename or entry.get("filename") or "unknown"

    if new_data:
        data = new_data
    elif entry.get("stored_path"):
        data = get_stored_file_bytes(entry["stored_path"])
    else:
        raise ValueError(
            f"Cannot re-index document_id={document_id!r}: no replacement content "
            "was given and no original file is available in object storage "
            "(it may have been uploaded while object storage was unavailable)."
        )

    fhash = _file_hash(data)
    stored_path = _save_uploaded_file(filename, data, fhash)

    docs = _load_document_from_bytes(filename, data)
    for d in docs:
        d.metadata["stored_path"] = stored_path
        # Document-isolation metadata (see update_db_files) — document_id
        # is intentionally the SAME as before, not regenerated.
        d.metadata["conversation_id"] = conversation_id
        d.metadata["document_id"] = document_id

    docs = _deduplicate(docs)
    if not docs:
        raise RuntimeError(
            f"Re-indexing document_id={document_id!r} extracted no content from "
            f"'{filename}' — refusing to replace the existing index with nothing."
        )

    _report("chunking")

    chunks = _chunk_documents(docs)
    if not chunks:
        raise RuntimeError(
            f"Re-indexing document_id={document_id!r} produced 0 chunks from "
            f"'{filename}' — refusing to replace the existing index with nothing."
        )

    cps: Counter = Counter(c.metadata.get("source", "?") for c in chunks)
    idx_map: Counter = Counter()
    for chunk in chunks:
        src = chunk.metadata.get("source", "?")
        chunk.metadata["chunk_index"] = idx_map[src]
        chunk.metadata["total_chunks"] = cps[src]
        idx_map[src] += 1

    _report("embedding")

    ensure_collection(embeddings)
    vdb = _get_vector_db()

    # New chunks are already validated non-empty above — safe to swap now:
    # delete the OLD version's points, then insert the NEW ones, with
    # nothing else happening in between.
    if is_ready():
        doc_filter = _document_filter(document_id, conversation_id)
        with_retries(
            lambda: get_client().delete(
                collection_name=get_collection_name(), points_selector=doc_filter
            ),
            what=f"delete stale vectors for document_id={document_id!r} before reindex",
        )

    vdb.add_documents(chunks)
    _refresh_retriever()

    registry = _load_registry()
    if old_key in registry:
        del registry[old_key]
    new_key = _registry_key(conversation_id, fhash)
    registry[new_key] = {
        "filename": filename,
        "stored_path": stored_path,
        "file_type": _get_file_type(filename),
        "chunks": len(chunks),
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "conversation_id": conversation_id,
        "document_id": document_id,
    }
    _save_registry(registry)

    log.info(f"Re-indexed document_id={document_id!r} ('{filename}') -> {len(chunks)} chunks.")

    _report("done")

    return len(chunks)


# ── Public API: direct question answering (non-agent /chat path) ───────────────

def ask_question(query: str, conversation_id: str, lang: str = "auto") -> dict:
    """
    Answer a question using RAG directly (single retrieval + single generation),
    scoped to `conversation_id`'s own uploaded documents only.
    Returns {"answer": str, "sources": str}.
    """
    if not is_ready():
        return {
            "answer": "⚠️ Database is empty. Please upload files first.",
            "sources": "",
        }

    detected_lang = detect_language(query) if lang == "auto" else lang
    docs, debug, _ = _retrieve(query, detected_lang, conversation_id)
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


def _trim_to_budget(ranked_texts_and_items: List[Tuple[str, Any]], max_chars: int) -> List[Any]:
    """
    Given (text, item) pairs already in rank order (best first), drop
    lowest-ranked items from the tail until the combined text length fits
    within `max_chars`. Always keeps at least the first (best) item, even
    if it alone exceeds the budget. Shared by both the Document-based
    context builder (`_build_context`, used by `ask_question`) and the
    agent's dict-based context builders (`generate_answer`, `summarize`,
    `compare`), so context-size trimming lives in one place.
    """
    total = 0
    keep: List[Any] = []
    for text, item in ranked_texts_and_items:
        length = len(text)
        if keep and total + length > max_chars:
            break
        total += length
        keep.append(item)
    return keep


def _chunk_label(meta: dict, index: int) -> str:
    """
    Build the '[Chunk N | ...]' header shown to the LLM for one retrieved
    chunk. Excel-sourced chunks (row_group/sheet_summary, see
    loaders/excel_loader.py) get a sheet/row-range header instead of the
    generic page number — without this, a table row looks like anonymous
    prose to the model and it has no signal it's reading structured data.
    Accepts either a Document.metadata dict (file_type) or the agent-facing
    dict shape from retrieve() (document_type) — both are plain dicts.
    """
    source = meta.get("source", "?")
    file_type = meta.get("document_type") or meta.get("file_type") or ""

    if file_type == "excel":
        sheet = meta.get("sheet_name")
        chunk_type = meta.get("chunk_type")
        row_range = meta.get("row_range")

        if chunk_type == "sheet_summary":
            detail = f"Sheet: {sheet} (summary)" if sheet else "Sheet summary"
        elif row_range:
            detail = f"Sheet: {sheet} | Rows {row_range}" if sheet else f"Rows {row_range}"
        elif sheet:
            detail = f"Sheet: {sheet}"
        else:
            detail = "Excel sheet"

        return f"[Chunk {index} | {source} | {detail}]"

    return f"[Chunk {index} | {source} | page {meta.get('page', 0)}]"


def _build_context(docs: List[Document]) -> str:
    budgeted = _trim_to_budget([(d.page_content, d) for d in docs], settings.MAX_CONTEXT_CHARS)
    context_parts = [
        f"{_chunk_label(d.metadata, i+1)}\n{d.page_content}"
        for i, d in enumerate(budgeted)
    ]
    return "\n\n---\n\n".join(context_parts)


# ── Public API used by the agent's tools ────────────────────────────────────────
#
# These functions wrap the pipeline above in a stable, agent-friendly shape
# (plain dicts instead of langchain Document objects) so the agent package
# never has to know about langchain/Qdrant internals.

def retrieve(
    question: str,
    conversation_id: str,
    lang: str = "auto",
    top_k: int = 5,
    source_filter: Optional[str] = None,
    raw_question: Optional[str] = None,
) -> List[dict]:
    """
    Retrieve relevant chunks for a question, restricted to `conversation_id`'s
    own uploaded documents (see Document Isolation — every vector search
    goes through `_retrieve`'s Qdrant-side conversation_id filter; there is
    no unscoped/global search mode). Used by the agent's retrieve tool, and
    by topic-scoped report generation (services.report_service) with a
    larger top_k and, optionally, `source_filter` to search within one
    named document instead of that conversation's whole knowledge base.

    `raw_question`, if given, is the ORIGINAL user message this retrieval
    is answering (as opposed to `question`, which may be an LLM-planner
    reformulation of it) — see `_retrieve`'s docstring for why this is
    added as a deterministic extra variant.

    Returns a list of dicts: {"id", "text", "metadata": {...}}
    """
    if not is_ready():
        return []

    detected_lang = detect_language(question) if lang == "auto" else lang
    docs, debug, debugs = _retrieve(
        question, detected_lang, conversation_id,
        k=max(top_k, settings.RETRIEVER_K), top_n=top_k,
        source_filter=source_filter,
        raw_question=raw_question,
    )
    log.debug(f"[agent] Retrieval for '{question}' (conversation_id={conversation_id!r}):\n{debug}")

    scores_by_id = {id(d): dbg.get("score", 0.0) for d, dbg in zip(docs, debugs)}

    results = []
    for d in docs:
        source = d.metadata.get("source", "?")
        page = d.metadata.get("page", 0)
        chunk_index = d.metadata.get("chunk_index", 0)
        chunk_id = f"{source}::{page}::{chunk_index}::{hashlib.md5(d.page_content.encode('utf-8')).hexdigest()[:8]}"
        metadata = {
            "source": source,
            "title": source,
            "document_type": d.metadata.get("file_type", "unknown"),
            "page": page,
            "chunk_index": chunk_index,
            # Document-isolation metadata (see update_db_files) — forwarded
            # for debuggability; retrieval is already scoped by the Qdrant
            # filter above regardless of these being present in the result.
            "conversation_id": d.metadata.get("conversation_id"),
            "document_id": d.metadata.get("document_id"),
            # Relevance score (0-1) from reranking — lets callers that ask
            # for many candidates at once (e.g. topic-scoped report
            # generation) filter out weakly-relevant padding instead of
            # treating every one of top_k results as equally on-topic.
            "relevance_score": scores_by_id.get(id(d), 0.0),
        }
        # Excel-specific metadata (see loaders/excel_loader.py::make_meta) —
        # forwarded so downstream context-building (_chunk_label) and the
        # LLM prompt can tell a whole-sheet summary apart from a specific
        # row range, instead of seeing anonymous chunk text.
        for key in ("sheet_name", "chunk_type", "row_range", "total_sheets", "sheet_row_count"):
            if d.metadata.get(key) is not None:
                metadata[key] = d.metadata[key]
        results.append({"id": chunk_id, "text": d.page_content, "metadata": metadata})
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

    budgeted = _trim_to_budget([(d["text"], d) for d in documents], settings.MAX_CONTEXT_CHARS)
    context_parts = [
        f"{_chunk_label(d['metadata'], i+1)}\n{d['text']}"
        for i, d in enumerate(budgeted)
    ]
    context = "\n\n---\n\n".join(context_parts)

    if settings.LOG_RETRIEVAL_DEBUG:
        log.info(
            f"[retrieval-debug] final context sent to LLM ({len(context)} chars, "
            f"{len(budgeted)} chunks): {context[:2000]!r}"
        )

    prompt = build_prompt_with_memory(context, question, detected_lang, memory)

    # A Groq/network failure here previously came back as a normal-looking
    # `answer` string ("Error generating answer: ..."), which routes/chat.py
    # would then return as a 200 OK chat response and Agent._remember would
    # persist into long-term memory as if it were a real assistant turn.
    # Let it propagate instead — routes/chat.py and routes/ws.py already
    # turn an uncaught exception into a proper error response/event, and it
    # never pollutes conversation memory. See the agent/llm.py fix for the
    # same reasoning applied to the planner's LLM call.
    with timing.stage("llm_generation"):
        answer = str(llm.invoke(prompt))
    return _clean_answer(answer, detected_lang)


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

    budgeted = _trim_to_budget([(d["text"], d) for d in documents], settings.MAX_CONTEXT_CHARS)
    context_parts = [
        f"{_chunk_label(d['metadata'], i+1)}\n{d['text']}"
        for i, d in enumerate(budgeted)
    ]
    context = "\n\n---\n\n".join(context_parts)

    prompt = build_prompt_with_memory(context, question, detected_lang, memory)

    # See generate_answer's comment above — a mid-stream API/network
    # failure should propagate (routes/ws.py's _produce() already turns an
    # uncaught exception from agent.run_stream() into a proper {"type":
    # "error"} event) rather than being yielded as if it were real answer
    # text the user would see appended to whatever streamed so far.
    yield from llm.stream(prompt)


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

    yield from llm.stream(prompt)


def summarize_stream(documents: List[dict], lang: str = "en"):
    """Streaming counterpart of `summarize`."""
    if not documents:
        yield "لا توجد مستندات لتلخيصها." if lang == "ar" else "No documents found to summarize."
        return

    budgeted = _trim_to_budget([(d["text"], d) for d in documents], settings.MAX_CONTEXT_CHARS)
    document_text = wrap_untrusted_context("\n\n".join(
        f"{_chunk_label(d['metadata'], i+1)}\n{d['text']}" for i, d in enumerate(budgeted)
    ), lang)
    prompt = (
        f"لخّص المستندات التالية بإيجاز ووضوح، بالاعتماد فقط على ما ورد فيها حرفيًا، دون إضافة أي معلومة من خارجها. كن مباشرًا؛ لا تضف مثالًا إلا إذا كان يوضّح نقطة غير بديهية:\n\n{document_text}"
        if lang == "ar"
        else f"Summarize the following documents clearly and concisely, using only what is stated in them — do not add any information from outside them. Be direct; only include an example if it clarifies a non-obvious point:\n\n{document_text}"
    )

    yield from llm.stream(prompt)


def compare_stream(question: str, documents: List[dict], lang: str = "en"):
    """Streaming counterpart of `compare`."""
    if not documents:
        yield "لا توجد مستندات كافية للمقارنة." if lang == "ar" else "No documents found to compare."
        return

    budgeted = _trim_to_budget([(d["text"], d) for d in documents], settings.MAX_CONTEXT_CHARS)
    document_text = wrap_untrusted_context("\n\n".join(
        f"{_chunk_label(d['metadata'], i+1)}\n{d['text']}" for i, d in enumerate(budgeted)
    ), lang)
    prompt = (
        f"باستخدام المستندات التالية حصريًا (لا تستخدم أي معرفة خارجية)، أجب عن طلب المقارنة بشكل مباشر ومنظم. إذا كانت المستندات لا تحتوي على معلومات كافية للمقارنة المطلوبة، قل ذلك صراحة:\n\nالسؤال:\n{question}\n\nالمستندات:\n\n{document_text}"
        if lang == "ar"
        else f"Using only the following documents (no outside knowledge), answer this comparison request directly and in a well-organized way. If the documents do not contain enough information for the requested comparison, say so explicitly:\n\nQuestion:\n{question}\n\nDocuments:\n\n{document_text}"
    )

    yield from llm.stream(prompt)


def answer_from_memory(question: str, memory: str, lang: str = "auto") -> str:
    """
    Answer a question using only conversation memory (no document retrieval).
    Strictly scoped to greetings/small talk/meta-conversation — see
    `_memory_only_prompt` for the full rule set. Used by the 'respond' tool
    and by 'generate' when no documents were retrieved but memory exists.
    """
    detected_lang = detect_language(question) if lang == "auto" else lang
    prompt = _memory_only_prompt(question, memory, detected_lang)

    # See generate_answer's comment above — let a real API/network failure
    # propagate rather than returning it as if it were a normal answer.
    answer = str(llm.invoke(prompt))
    return _clean_answer(answer, detected_lang)


def summarize(documents: List[dict], lang: str = "en") -> str:
    """Summarize a set of agent-collected documents. Used by the summarize tool."""
    if not documents:
        return "لا توجد مستندات لتلخيصها." if lang == "ar" else "No documents found to summarize."

    budgeted = _trim_to_budget([(d["text"], d) for d in documents], settings.MAX_CONTEXT_CHARS)
    document_text = wrap_untrusted_context("\n\n".join(
        f"{_chunk_label(d['metadata'], i+1)}\n{d['text']}" for i, d in enumerate(budgeted)
    ), lang)

    if lang == "ar":
        prompt = f"لخّص المستندات التالية بإيجاز ووضوح، بالاعتماد فقط على ما ورد فيها حرفيًا، دون إضافة أي معلومة من خارجها. كن مباشرًا؛ لا تضف مثالًا إلا إذا كان يوضّح نقطة غير بديهية:\n\n{document_text}"
    else:
        prompt = f"Summarize the following documents clearly and concisely, using only what is stated in them — do not add any information from outside them. Be direct; only include an example if it clarifies a non-obvious point:\n\n{document_text}"

    return _clean(str(llm.invoke(prompt)))


def compare(question: str, documents: List[dict], lang: str = "en") -> str:
    """Compare information across agent-collected documents. Used by the compare tool."""
    if not documents:
        return "لا توجد مستندات كافية للمقارنة." if lang == "ar" else "No documents found to compare."

    budgeted = _trim_to_budget([(d["text"], d) for d in documents], settings.MAX_CONTEXT_CHARS)
    document_text = wrap_untrusted_context("\n\n".join(
        f"{_chunk_label(d['metadata'], i+1)}\n{d['text']}" for i, d in enumerate(budgeted)
    ), lang)

    if lang == "ar":
        prompt = f"باستخدام المستندات التالية حصريًا (لا تستخدم أي معرفة خارجية)، أجب عن طلب المقارنة بشكل مباشر ومنظم. إذا كانت المستندات لا تحتوي على معلومات كافية للمقارنة المطلوبة، قل ذلك صراحة:\n\nالسؤال:\n{question}\n\nالمستندات:\n\n{document_text}"
    else:
        prompt = f"Using only the following documents (no outside knowledge), answer this comparison request directly and in a well-organized way. If the documents do not contain enough information for the requested comparison, say so explicitly:\n\nQuestion:\n{question}\n\nDocuments:\n\n{document_text}"

    return _clean(str(llm.invoke(prompt)))


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
