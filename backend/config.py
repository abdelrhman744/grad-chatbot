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


class Settings:
    # ── LLM (Groq) ───────────────────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Model used by the agent's action-selection step. Defaults to the same
    # model as the main answer generator; override with AGENT_MODEL to use
    # a smaller/faster model (e.g. "llama-3.1-8b-instant") for planning.
    AGENT_MODEL: str = os.getenv("AGENT_MODEL", GROQ_MODEL)

    LLM_TEMPERATURE: float = _float("LLM_TEMPERATURE", 0.0)
    LLM_MAX_TOKENS: int = _int("LLM_MAX_TOKENS", 800)
    LLM_TOP_P: float = _float("LLM_TOP_P", 0.90)

    # ── Embeddings ───────────────────────────────────────────────────────
    # Local sentence-transformers model — runs on-device, no API key, no
    # embedding API calls. See services/embeddings_provider.py.
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

    # ── Vector store (Qdrant, local/embedded) ───────────────────────────
    QDRANT_PATH: str = os.getenv("QDRANT_PATH", "./qdrant_db")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "enterprise_docs")

    # ── RAG pipeline ─────────────────────────────────────────────────────
    # Deprecated: originals are now stored in MinIO (MINIO_BUCKET_UPLOADS).
    # Kept only so old .env files with this var set don't error out.
    UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", "./stored_files")
    PROCESSED_FILES_REGISTRY: str = os.getenv(
        "PROCESSED_FILES_REGISTRY", "./processed_files.json"
    )
    ENABLE_PDF_OCR_FALLBACK: bool = _bool("ENABLE_PDF_OCR_FALLBACK", True)
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
    CHUNK_SIZE: int = _int("CHUNK_SIZE", 700)
    CHUNK_OVERLAP: int = _int("CHUNK_OVERLAP", 150)

    # ── Audio / OCR ──────────────────────────────────────────────────────
    WHISPER_MODEL_NAME: str = os.getenv("WHISPER_MODEL_NAME", "small")
    SILENCE_THRESHOLD_DB: float = _float("SILENCE_THRESHOLD_DB", -60.0)
    FFMPEG_PATH: str = os.getenv("FFMPEG_PATH", "ffmpeg")
    TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", "tesseract")

    # ── Agent ────────────────────────────────────────────────────────────
    AGENT_MAX_ITERATIONS: int = _int("AGENT_MAX_ITERATIONS", 6)
    AGENT_DEBUG: bool = _bool("AGENT_DEBUG", False)
    DEFAULT_CONVERSATION_ID: str = os.getenv("DEFAULT_CONVERSATION_ID", "default")

    # ── Memory ───────────────────────────────────────────────────────────
    MEMORY_MAX_MESSAGES: int = _int("MEMORY_MAX_MESSAGES", 25)
    MEMORY_KEEP_RECENT: int = _int("MEMORY_KEEP_RECENT", 4)
    MEMORY_WINDOW: int = _int("MEMORY_WINDOW", 6)
    MEMORY_STORAGE_DIR: str = os.getenv("MEMORY_STORAGE_DIR", "./memory_storage")

    # ── MinIO (object storage for uploaded files & generated reports) ─────
    # Uploaded originals are no longer written to local disk — they are
    # streamed straight into a MinIO bucket. Set these to point at your own
    # MinIO deployment (see docker-compose.yml for a local dev instance).
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
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
