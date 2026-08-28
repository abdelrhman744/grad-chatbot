"""
config.py

Centralized application configuration.

All environment-driven settings live here so that services, the agent,
and memory modules never read `os.environ` directly. LLM generation runs
on the Groq API; embeddings run entirely locally via `sentence-transformers`
(no embedding API, no key) — see .env.example for every variable and its
default.
"""

from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    val = os.getenv(name)
    try:
        return int(val) if val is not None else default
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    val = os.getenv(name)
    try:
        return float(val) if val is not None else default
    except ValueError:
        return default


def _secret_key() -> str:
    val = os.getenv("APP_SECRET_KEY", "").strip()
    if val:
        return val
    generated = secrets.token_hex(32)
    print(
        "WARNING: APP_SECRET_KEY is not set - generated a random key for this "
        "process only. Every conversation token signed with it stops verifying "
        "the moment this process restarts, which means every currently open "
        "conversation becomes inaccessible (its documents/memory are NOT "
        "deleted, just no longer reachable with the old token). Set "
        "APP_SECRET_KEY to a fixed, persistent value in your .env for anything "
        "beyond throwaway local testing.",
        file=sys.stderr,
    )
    return generated


class Settings:
    # ── Security ─────────────────────────────────────────────────────────
    # Signs conversation_id tokens (see utils/conversation_auth.py) so a
    # bare, guessed/leaked conversation_id string is no longer enough to
    # read/modify/delete someone else's documents or memory — every
    # request must also present a signature that only this key can
    # produce. See _secret_key()'s warning above for what happens if this
    # is left unset.
    APP_SECRET_KEY: str = _secret_key()

    # ── CORS ─────────────────────────────────────────────────────────────
    # Comma-separated allowlist of origins permitted to call the API with
    # credentials. Wildcard ("*") is intentionally not supported here: it
    # is rejected by browsers when combined with allow_credentials=True
    # (see main.py), so a real allowlist is required. Defaults to the
    # frontend's own dev/Docker origin.
    FRONTEND_ORIGIN: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")

    # ── LLM (Groq) ───────────────────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Model used by the agent's action-selection step. This is a structured
    # JSON routing decision (pick one of a handful of tools), not open-ended
    # generation, so a smaller/faster model is used by default to cut
    # per-turn planner latency; override with AGENT_MODEL (e.g. back to
    # GROQ_MODEL) if routing quality needs the larger model instead.
    AGENT_MODEL: str = os.getenv("AGENT_MODEL", "llama-3.1-8b-instant")

    # Every JSON-mode model occasionally has Groq's own server-side JSON
    # validator reject a request outright (400 json_validate_failed /
    # json_generate_failed) -- confirmed happening for AGENT_MODEL itself
    # during live testing (see agent/llm.py). Unlike a malformed-but-present
    # response, there is no output to ask the model to self-correct, and
    # unlike a 429/network/auth failure this is specific to one particular
    # generation attempt -- retrying the identical prompt against a
    # DIFFERENT model is the one case where papering over the failure is
    # safe rather than masking a real problem. Deliberately a distinct
    # model from AGENT_MODEL (see agent/llm.py's AgentLLM) so a systemic
    # issue with the primary model's pool (rate limit, outage) doesn't take
    # the fallback down with it.
    AGENT_FALLBACK_MODEL: str = os.getenv("AGENT_FALLBACK_MODEL", "qwen/qwen3.8-27b")

    LLM_TEMPERATURE: float = _float("LLM_TEMPERATURE", 0.0)
    LLM_MAX_TOKENS: int = _int("LLM_MAX_TOKENS", 800)
    LLM_TOP_P: float = _float("LLM_TOP_P", 0.90)

    # The Groq SDK retries a failed request (rate limits, transient 5xx)
    # internally, honoring the API's own Retry-After hint, before raising.
    # A single /api/chat turn can make several sequential/concurrent Groq
    # calls (query translation, query rewrite, multiple agent-planning
    # iterations, final generation) — if the account's tokens-per-minute
    # quota is tight, each of those calls can independently hit this retry
    # path, and the backoffs stack into a multi-tens-of-seconds-to-minutes
    # wall-clock delay for one user-visible request. Made explicit and
    # configurable here (rather than left at the SDK's internal default) so
    # this ceiling is visible and tunable instead of an invisible source of
    # the "hangs, then times out" symptom. See services/llm_provider.py.
    GROQ_MAX_RETRIES: int = _int("GROQ_MAX_RETRIES", 2)
    # Hard per-call ceiling so a single Groq request can never hang
    # indefinitely regardless of retry configuration.
    GROQ_REQUEST_TIMEOUT_SECONDS: float = _float("GROQ_REQUEST_TIMEOUT_SECONDS", 30.0)

    # ── Embeddings ───────────────────────────────────────────────────────
    # Local sentence-transformers model — runs on-device, no API key, no
    # embedding API calls. See services/embeddings_provider.py.
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    # Device for BOTH local torch models (the embedding model above and the
    # cross-encoder reranker below): "auto" (default) uses a CUDA GPU when
    # one is available and falls back to CPU otherwise — safe on machines
    # with no GPU or a CPU-only torch build. Set "cpu" or "cuda" to force a
    # choice. See utils/device.py.
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "auto")

    # ── Vector store (Qdrant, server mode) ──────────────────────────────
    # Points at a Qdrant *server* (REST API) — either the `qdrant` service
    # in docker-compose.yml (QDRANT_URL=http://qdrant:6333, set via the
    # backend service's `environment:` block) or a locally-run Qdrant
    # server for native (non-Docker) development. Embedded/file-mode
    # Qdrant is no longer supported.
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "enterprise_docs")
    QDRANT_TIMEOUT_SECONDS: float = _float("QDRANT_TIMEOUT_SECONDS", 10.0)
    # Bounded retry/backoff applied to Qdrant operations (client creation,
    # collection checks, deletes, ...) so a transient network blip or a
    # Qdrant container that's still starting doesn't immediately fail the
    # request/startup — see services/db_service.py::_with_retries.
    QDRANT_CONNECT_RETRIES: int = _int("QDRANT_CONNECT_RETRIES", 5)
    QDRANT_RETRY_DELAY_SECONDS: float = _float("QDRANT_RETRY_DELAY_SECONDS", 2.0)

    # ── RAG pipeline ─────────────────────────────────────────────────────
    # Deprecated: originals are now stored in MinIO (MINIO_BUCKET_UPLOADS).
    # Kept only so old .env files with this var set don't error out.
    UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", "./stored_files")
    PROCESSED_FILES_REGISTRY: str = os.getenv(
        "PROCESSED_FILES_REGISTRY", "./processed_files.json"
    )
    ENABLE_PDF_OCR_FALLBACK: bool = _bool("ENABLE_PDF_OCR_FALLBACK", True)
    # Hard cap on a single uploaded file's size. Rejected upfront with a
    # clear error rather than letting an arbitrarily large file run through
    # a fully in-memory, fully synchronous parse/chunk/embed pipeline.
    MAX_UPLOAD_SIZE_MB: int = _int("MAX_UPLOAD_SIZE_MB", 200)
    RETRIEVER_K: int = _int("RETRIEVER_K", 8)
    RERANK_TOP_N: int = _int("RERANK_TOP_N", 6)
    # Coarse pre-filter only (see _retrieve() in rag_service.py) — lexical
    # overlap score below which retrieved chunks are discarded entirely as
    # "no relevant match". Kept low deliberately: testing showed this
    # lexical heuristic cannot reliably separate on-topic from off-topic
    # questions on its own (a loosely-worded but genuine question can score
    # similarly to an unrelated one), so it only catches near-zero-overlap
    # cases. The real grounding guard is the LLM prompt rule in
    # build_prompt() that refuses to answer unless the context specifically
    # covers the question.
    CONFIDENCE_THRESHOLD: float = _float("CONFIDENCE_THRESHOLD", 0.05)

    # ── Reranking (cross-encoder + lexical blend, diversity, context budget) ──
    # Cross-encoder reranking blends a semantic relevance score (a small
    # multilingual cross-encoder model, CPU-friendly, no API key — same
    # "local, free" philosophy as the embeddings model) with the existing
    # lexical/bigram overlap score, instead of relying on lexical overlap
    # alone. Falls back permanently to lexical-only scoring for the process
    # lifetime if the model can't be loaded (offline, etc).
    RERANK_USE_CROSS_ENCODER: bool = _bool("RERANK_USE_CROSS_ENCODER", True)
    CROSS_ENCODER_MODEL: str = os.getenv(
        "CROSS_ENCODER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    )
    # When any retrieved candidate is Excel-sourced (row_group/sheet_summary
    # chunks), _retrieve() widens the final reranked result count up to this
    # many chunks instead of stopping at RERANK_TOP_N — a question needing
    # several spreadsheet rows would otherwise lose most of them to the
    # smaller default top_n. See _retrieve() in rag_service.py.
    EXCEL_RERANK_TOP_N: int = _int("EXCEL_RERANK_TOP_N", 12)
    # Weight given to the cross-encoder score vs. the lexical score when both
    # are available (0-1, higher = trust the cross-encoder more).
    RERANK_ALPHA: float = _float("RERANK_ALPHA", 0.6)
    # After scoring, greedily reselect the top_n chunks to also maximize
    # diversity (MMR-lite) so near-duplicate/overlapping chunks don't crowd
    # out distinct information within the same context window.
    RERANK_DIVERSIFY: bool = _bool("RERANK_DIVERSIFY", True)
    MMR_LAMBDA: float = _float("MMR_LAMBDA", 0.7)
    # Hard cap on how many characters of retrieved context are sent to the
    # LLM per answer — bounds prompt token usage regardless of how many/how
    # large the reranked chunks are. Lowest-ranked chunks are trimmed first.
    MAX_CONTEXT_CHARS: int = _int("MAX_CONTEXT_CHARS", 6000)
    CHUNK_SIZE: int = _int("CHUNK_SIZE", 700)
    CHUNK_OVERLAP: int = _int("CHUNK_OVERLAP", 150)

    # Query expansion: adds LLM-generated synonym/concept reformulations of the
    # user's ORIGINAL-language query (e.g. "advantages" -> "benefits, pros") as
    # extra retrieval variants, on top of the existing translation/typo-fix/
    # rephrase variants. Helps semantic/evaluative questions ("which is more
    # efficient?", "pros and cons?") match document wording that doesn't share
    # the user's exact vocabulary. See _rewrite_query() / _query_variants().
    QUERY_EXPANSION_ENABLED: bool = _bool("QUERY_EXPANSION_ENABLED", True)

    # Chunking strategy: "recursive" (default, unchanged behavior — fixed
    # character-size splitting via RecursiveCharacterTextSplitter), "semantic"
    # (per-sentence embedding windows — most precise boundaries, but slowest:
    # one embedding call per sentence), or "hybrid" (fast recursive splitting
    # into small base chunks, then ONE batched embedding call over those base
    # chunks — far fewer vectors than "semantic" — merging adjacent chunks
    # that are semantically similar). "hybrid" is the recommended
    # speed/quality tradeoff if you want semantic-aware boundaries without
    # the per-sentence embedding cost.
    CHUNKING_STRATEGY: str = os.getenv("CHUNKING_STRATEGY", "recursive")
    # Number of sentences grouped together before embedding (bigger buffer
    # = fewer embedding calls = faster, at the cost of coarser boundaries).
    SEMANTIC_CHUNK_BUFFER_SIZE: int = _int("SEMANTIC_CHUNK_BUFFER_SIZE", 3)
    # Percentile of sentence-to-sentence distance jumps used as the
    # split-point threshold (higher = fewer, larger chunks).
    SEMANTIC_CHUNK_BREAKPOINT_PERCENTILE: float = _float(
        "SEMANTIC_CHUNK_BREAKPOINT_PERCENTILE", 90.0
    )
    # Safety cap: a semantic chunk larger than this many characters is
    # further split with the recursive splitter so downstream chunk-size
    # assumptions (retrieval, prompt building) never break.
    SEMANTIC_CHUNK_MAX_CHARS: int = _int("SEMANTIC_CHUNK_MAX_CHARS", 1800)

    # ── Hybrid chunking (fast semantic-aware merging) ──────────────────────
    # Base chunk size fed into the first (recursive, cheap) pass. Smaller =
    # finer-grained merge decisions but more vectors to embed.
    HYBRID_BASE_CHUNK_SIZE: int = _int("HYBRID_BASE_CHUNK_SIZE", 300)
    # Cosine-similarity threshold above which two adjacent base chunks are
    # merged into one (0-1, higher = stricter = more, smaller chunks).
    HYBRID_MERGE_SIMILARITY_THRESHOLD: float = _float(
        "HYBRID_MERGE_SIMILARITY_THRESHOLD", 0.62
    )
    # Safety cap: merged chunks never grow past this many characters.
    HYBRID_CHUNK_MAX_CHARS: int = _int("HYBRID_CHUNK_MAX_CHARS", 1800)

    # ── Excel ingestion (.xlsx/.xls/.csv) ───────────────────────────────
    # Row-group chunk size is derived from CHUNK_SIZE/CHUNK_OVERLAP above
    # (see loaders/excel_loader.py::_rows_per_group); these only bound how
    # few/many rows a single chunk can hold.
    EXCEL_ROWS_PER_CHUNK_MIN: int = _int("EXCEL_ROWS_PER_CHUNK_MIN", 3)
    EXCEL_ROWS_PER_CHUNK_MAX: int = _int("EXCEL_ROWS_PER_CHUNK_MAX", 50)
    EXCEL_SUMMARY_SAMPLE_ROWS: int = _int("EXCEL_SUMMARY_SAMPLE_ROWS", 5)
    # Sheets larger than this are truncated (for indexing only) so a
    # pathologically large spreadsheet can't block the synchronous upload
    # request indefinitely.
    EXCEL_MAX_ROWS_PER_SHEET: int = _int("EXCEL_MAX_ROWS_PER_SHEET", 20000)

    # ── Audio / OCR ──────────────────────────────────────────────────────
    WHISPER_MODEL_NAME: str = os.getenv("WHISPER_MODEL_NAME", "small")
    SILENCE_THRESHOLD_DB: float = _float("SILENCE_THRESHOLD_DB", -60.0)
    FFMPEG_PATH: str = os.getenv("FFMPEG_PATH", "ffmpeg")
    TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", "tesseract")
    # Bounded page-level parallelism for Tesseract OCR (services/ocr_service.py).
    # Each concurrent page spawns its own Tesseract subprocess, so this
    # directly caps how many Tesseract processes can run at once — keep it
    # modest on shared/weak servers regardless of page count or CPU count.
    OCR_MAX_CONCURRENT_PAGES: int = _int("OCR_MAX_CONCURRENT_PAGES", 4)
    # A page/image's first (cheapest) OCR attempt is accepted as-is once its
    # extracted text clears this length — only then does the full
    # preprocessing-strategy x PSM-mode sweep run (see
    # ocr_service._ocr_image_tiered). Same value PyPDFLoader's own
    # whole-document text-length check already used (unchanged threshold,
    # now also reused per-page — see loaders/pdf_loader.py).
    OCR_MIN_TEXT_CHARS: int = _int("OCR_MIN_TEXT_CHARS", 20)
    # Minimum fraction of non-whitespace characters that must be
    # alphanumeric (Arabic or Latin letters, digits) for a first-attempt
    # OCR result to be trusted without escalating to the full sweep — cheap
    # guard against accepting sparse/garbled noise from a single pass.
    OCR_MIN_ALNUM_RATIO: float = _float("OCR_MIN_ALNUM_RATIO", 0.6)

    # ── Handwritten OCR (TrOCR, local via Hugging Face `transformers`) ────
    # Free/local/offline-capable handwriting recognition — separate from the
    # printed-text OCR above (Tesseract, still used for scanned PDFs/images
    # in the upload pipeline). See services/handwritten_ocr_service.py.
    # Both models are downloaded automatically by `transformers` on first
    # use and cached under the standard Hugging Face cache dir (~/.cache/
    # huggingface — already persisted by docker-compose.yml's
    # `backend_model_cache` volume, same as the embedding/Whisper models);
    # no manual download and no local path hardcoding.
    #
    # English default changed base -> small after evaluate_handwritten_ocr.py
    # (Tasks 2/6): on real IAM handwriting-line samples, run through the
    # SAME preprocessing pipeline, trocr-small-handwritten matched
    # trocr-base-handwritten's accuracy (CER 0.253 vs 0.248 — within noise
    # over a 6-sample benchmark) at 3-6x lower CPU per-line latency and a
    # much smaller checkpoint (~62M vs ~334M params) — a clear win with no
    # measured accuracy cost for the "weak university servers, no GPU"
    # target environment. See the final report for the full evidence table.
    HANDWRITTEN_OCR_EN_MODEL: str = os.getenv(
        "HANDWRITTEN_OCR_EN_MODEL", "microsoft/trocr-small-handwritten"
    )
    # No realistic lighter Arabic alternative was found (searched the HF
    # Hub — this remains the only free/local Arabic handwriting checkpoint
    # identified); kept as-is. Its real-handwriting accuracy is genuinely
    # poor even after the line-segmentation aspect-ratio fix (see the final
    # report's OCR Decision / Remaining Issues) — a known, now-quantified
    # limitation, not something this task's scope (no huge VLM, no GPU
    # assumption) can fully resolve.
    HANDWRITTEN_OCR_AR_MODEL: str = os.getenv(
        "HANDWRITTEN_OCR_AR_MODEL", "RayR1/trocr-base-arabic-handwritten"
    )
    # Upper bound on generated tokens per OCR call — TrOCR models target
    # single text-line images, so this only needs to comfortably cover one
    # line/short passage, not a full page.
    HANDWRITTEN_OCR_MAX_NEW_TOKENS: int = _int("HANDWRITTEN_OCR_MAX_NEW_TOKENS", 256)
    # Multi-line pages are batched through the model this many lines at a
    # time (instead of one call per line) — see
    # HandwrittenOCRService._recognize_lines / scripts/evaluate_ocr_followup.py
    # for the benchmark this default is based on. Bounds peak RAM growth
    # for a page with many lines (up to _LINE_MAX_COUNT=80) instead of
    # batching the whole page in one call.
    HANDWRITTEN_OCR_MAX_BATCH_SIZE: int = _int("HANDWRITTEN_OCR_MAX_BATCH_SIZE", 8)

    # ── Agent ────────────────────────────────────────────────────────────
    AGENT_MAX_ITERATIONS: int = _int("AGENT_MAX_ITERATIONS", 6)
    AGENT_DEBUG: bool = _bool("AGENT_DEBUG", False)
    DEFAULT_CONVERSATION_ID: str = os.getenv("DEFAULT_CONVERSATION_ID", "default")

    # ── Agent lifecycle (in-process agent/session.py registry) ─────────────
    # A conversation's Agent — and the ShortMemory/FactStore/active_document
    # it owns in RAM — is evicted from the registry once it has had no
    # in-flight request AND has been inactive for this many seconds. A
    # later request with the same conversation_id transparently creates a
    # fresh Agent; only the in-RAM state is affected, not the persisted
    # memory_storage/*.json fact store. Default 1800s = 30 minutes.
    AGENT_IDLE_TIMEOUT_SECONDS: int = _int("AGENT_IDLE_TIMEOUT_SECONDS", 1800)
    # How often the background cleanup pass scans the registry for idle
    # conversations to evict. A conversation can live up to roughly
    # (AGENT_IDLE_TIMEOUT_SECONDS + AGENT_CLEANUP_INTERVAL_SECONDS) after
    # its last activity before it's actually evicted, in the worst case.
    # Default 300s = 5 minutes — a full dict scan at this size is
    # microseconds of work even with tens of thousands of entries, so this
    # interval is set for freshness, not to reduce scan cost.
    AGENT_CLEANUP_INTERVAL_SECONDS: int = _int("AGENT_CLEANUP_INTERVAL_SECONDS", 300)

    # ── Profiling / debugging ───────────────────────────────────────────────
    # Logs a per-stage latency breakdown (see utils/timing.py) for every
    # /api/chat request: language detection, query rewriting/translation,
    # embedding, Qdrant retrieval, reranking, MMR, agent planning, memory,
    # LLM generation, and total request time. Overhead is a handful of
    # perf_counter() calls per request — safe to leave on in production.
    LOG_REQUEST_PROFILE: bool = _bool("LOG_REQUEST_PROFILE", True)
    # Python logging level for the whole app (main.py's logging.basicConfig).
    # Kept at INFO by default — deliberately separate from AGENT_DEBUG /
    # LOG_RETRIEVAL_DEBUG, which gate WHETHER certain log.debug(...) calls
    # exist in a given code path at all; this controls whether debug-level
    # calls emit anything regardless. Only needs to be "DEBUG" when actively
    # tracing (see agent.py's _debug_step, llm_provider.py's
    # _log_outgoing_messages).
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    # Logs original query, detected language, every generated query variant,
    # retrieved chunks per variant, lexical/cross-encoder scores before and
    # after reranking, and the final context handed to the LLM. Verbose —
    # intended for debugging retrieval/cross-language issues, not normal
    # operation.
    LOG_RETRIEVAL_DEBUG: bool = _bool("LOG_RETRIEVAL_DEBUG", False)

    # ── Memory ───────────────────────────────────────────────────────────
    MEMORY_MAX_MESSAGES: int = _int("MEMORY_MAX_MESSAGES", 25)
    MEMORY_KEEP_RECENT: int = _int("MEMORY_KEEP_RECENT", 4)
    MEMORY_WINDOW: int = _int("MEMORY_WINDOW", 6)
    MEMORY_STORAGE_DIR: str = os.getenv("MEMORY_STORAGE_DIR", "./memory_storage")
    # Long-term memory is a capped, deduplicated store of discrete facts
    # (see memory/summary_memory.py's FactStore) rather than one free-text
    # paragraph. MEMORY_MAX_FACTS bounds how many facts are kept — lowest
    # importance/oldest are evicted first once exceeded. MEMORY_SUMMARY_MAX_CHARS
    # bounds how much of the rendered fact text is injected into any single
    # prompt, regardless of how many facts have accumulated.
    MEMORY_MAX_FACTS: int = _int("MEMORY_MAX_FACTS", 40)
    MEMORY_SUMMARY_MAX_CHARS: int = _int("MEMORY_SUMMARY_MAX_CHARS", 1200)
    # Short-term memory also summarizes once total character count crosses
    # this budget, even if MEMORY_MAX_MESSAGES hasn't been reached yet — a
    # handful of very long messages shouldn't be able to bloat token usage
    # before the message-count trigger fires.
    MEMORY_MAX_CHARS: int = _int("MEMORY_MAX_CHARS", 12000)

    # ── MinIO (object storage for uploaded files & generated reports) ─────
    # Uploaded originals are no longer written to local disk — they are
    # streamed straight into a MinIO bucket. Set these to point at your own
    # MinIO deployment (see docker-compose.yml for a local dev instance).
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    # Host:port used ONLY when building presigned download URLs — these are
    # handed to the browser, which cannot resolve the Docker service name
    # `minio` used for MINIO_ENDPOINT (the backend's own internal S3 calls).
    # Defaults to MINIO_ENDPOINT so native (non-Docker) dev, where both are
    # already the same host, needs no extra config; docker-compose.yml
    # overrides this to `localhost:9000` for the backend container.
    MINIO_PUBLIC_ENDPOINT: str = os.getenv("MINIO_PUBLIC_ENDPOINT", "") or os.getenv(
        "MINIO_ENDPOINT", "localhost:9000"
    )
    # minio-py needs a bucket's region to sign requests (incl. presigned
    # URLs) and otherwise auto-discovers it with a live GET request against
    # MINIO_ENDPOINT — which the MINIO_PUBLIC_ENDPOINT-configured signing
    # client can't reach (that host is meant for the browser, not the
    # backend container). Setting it explicitly skips that lookup entirely.
    # "us-east-1" is the minio/minio image's own default region.
    MINIO_REGION: str = os.getenv("MINIO_REGION", "us-east-1")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_SECURE: bool = _bool("MINIO_SECURE", False)
    MINIO_BUCKET_UPLOADS: str = os.getenv("MINIO_BUCKET_UPLOADS", "doc-assistant-uploads")
    MINIO_BUCKET_REPORTS: str = os.getenv("MINIO_BUCKET_REPORTS", "doc-assistant-reports")
    # How long a generated presigned download URL stays valid, in seconds.
    MINIO_PRESIGNED_EXPIRY: int = _int("MINIO_PRESIGNED_EXPIRY", 3600)

    # ── Report generation (per-document PDF summary) ───────────────────────
    REPORT_MAP_CHUNK_CHARS: int = _int("REPORT_MAP_CHUNK_CHARS", 6000)
    REPORT_FONT_DIR: str = os.getenv(
        "REPORT_FONT_DIR", str(BASE_DIR / "assets" / "fonts")
    )


settings = Settings()
