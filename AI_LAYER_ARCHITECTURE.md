# AI Layer Architecture Report

**Repository:** `grad-chatbot-Ibrahim_Hybrid` (branch `handwritten-ocr`)
**Scope:** Read-only investigation of the entire AI layer — backend agent/RAG/LLM stack, document ingestion, OCR (printed + handwritten), audio/STT, memory, and the frontend surfaces that participate in the pipeline.
**Method:** Every file listed below was read in full (not inferred from its name). Call sites were verified with `grep`/cross-reads across the whole `backend/` and `frontend/` trees. Where something could not be verified from source, it is stated explicitly rather than guessed.

---

## 1. Executive Summary

This is a bilingual (Arabic/English) **document Q&A assistant**: FastAPI backend + Next.js frontend, built around a **ReAct-style agent** (`backend/agent/agent.py`) that sits on top of a **RAG pipeline** (`backend/services/rag_service.py`). Documents (PDF, DOCX, TXT/MD, JSON, images, Excel/CSV) are uploaded, OCR'd if scanned, chunked, embedded locally (`sentence-transformers`), and stored in **Qdrant**. Chat happens over a streaming **WebSocket** (`/ws/chat`) or a plain HTTP endpoint (`/api/chat`), both driven by the same `Agent` class, which decides at each step whether to retrieve documents, generate/summarize/compare, respond from memory, or produce a PDF report. All LLM calls (main generation, agent planning, query rewriting/translation, memory fact-extraction, report writing) go through one provider, **Groq** — there is no local/self-hosted LLM. Embeddings and the cross-encoder reranker run entirely locally (no embedding API). Voice input is transcribed with **OpenAI Whisper** via a dedicated HTTP endpoint (not the WebSocket); there is **no text-to-speech** anywhere in the project. A separate, optional **handwritten-OCR** feature (Hugging Face TrOCR) exists behind its own route and is not part of the automatic upload pipeline. Long-term memory is a per-conversation, LLM-extracted, deduplicated **fact store** persisted to JSON on disk; short-term memory is an in-RAM message window. Object storage (originals, generated reports) is MinIO; everything is designed to degrade gracefully (Qdrant/MinIO retries, cross-encoder fallback, memory-extraction failures swallowed) rather than hard-fail a chat request.

---

## 2. Repository / AI Layer Structure

Real structure discovered (not assumed), AI-layer-relevant parts only:

```
backend/
├── agent/                     ReAct agent: planning loop, tools, prompts, schemas, session registry
│   ├── agent.py, llm.py, prompt.py, registry.py, schemas.py, session.py
│   └── tools/                 One class per terminal/non-terminal agent action
│       ├── retrieve_tool.py, generate_tool.py, summarize_tool.py,
│       │   compare_tool.py, respond_tool.py, report_tool.py
├── services/                  Core engines: RAG, LLM, embeddings, DB, OCR (x2), audio, memory glue, storage, reports
│   ├── rag_service.py, llm_provider.py, embeddings_provider.py, db_service.py
│   ├── ocr_service.py, handwritten_ocr_service.py, audio_service.py
│   ├── storage_service.py, upload_jobs.py, report_service.py
├── loaders/                   Per-file-type document parsers + dispatch registry
│   ├── registry.py, base.py, pdf_loader.py, docx_loader.py, text_loader.py,
│   │   image_loader.py, excel_loader.py
├── memory/                    Per-conversation short-term + long-term (fact store) memory
│   ├── short_memory.py, summary_memory.py, fact_extractor.py, llm_adapter.py, memory_manager.py
├── routes/                    FastAPI HTTP + WebSocket endpoints
│   ├── chat.py, ws.py, upload.py, ocr.py, reports.py, health.py
├── utils/                     Cross-cutting helpers
│   ├── device.py, timing.py, file_validation.py
├── scripts/                   Offline OCR evaluation harnesses (not part of runtime)
├── tests/                     Pytest suite (chunking, document lifecycle, file validation, handwritten OCR)
├── assets/fonts/               Amiri Arabic fonts for PDF report rendering
├── memory_storage/             Per-conversation JSON fact-store files (runtime data, not code)
├── config.py, main.py          Central settings + FastAPI app wiring
└── requirements.txt, .env.example, Dockerfile, HANDWRITTEN_OCR.md, PROFILING.md

frontend/
├── app/                        page.tsx (shell), layout.tsx
├── components/                 ChatBox, AnswerBox, SourceBox, UploadBox, VoiceRecorder,
│                                HandwrittenOcrModal, ReportCard, ui/ (Badge, Card, EmptyState, Skeleton)
├── lib/                        conversation.ts (conversation_id), fileTypeMeta.ts
└── services/api.ts             All backend I/O: REST + WebSocket client
```

`backend/tmp_unused/` exists but is an **empty directory** (no files) — not investigated further since it contains nothing.

---

## 3. Folder-by-Folder Explanation

### `backend/agent/`
**Responsibility:** The ReAct planning loop and its tool set. Owns one `Agent` instance per conversation, decides the next action (retrieve/generate/summarize/compare/respond/report), and streams or returns the final answer.
**Category:** Agent.
**Depends on:** `services/rag_service.py` (retrieval + generation), `services/llm_provider.py` (via `rag_service`), `memory/` (short+long-term memory), `utils/timing.py`.
**Depended on by:** `routes/chat.py`, `routes/ws.py` (both call `agent.session.get_agent`).

### `backend/agent/tools/`
**Responsibility:** One thin class per agent action, translating the planner's chosen action into a call into `rag_service`/`report_service`. Never touch Qdrant/LLM SDKs directly.
**Category:** Agent / RAG bridge.
**Depends on:** `services/rag_service.py`, `services/report_service.py`.
**Depended on by:** `agent/registry.py` → `agent/agent.py`.

### `backend/services/`
**Responsibility:** The actual engines — RAG (retrieval, reranking, generation, ingestion), the single Groq LLM wrapper, the local embedding model, Qdrant client, two independent OCR services (printed + handwritten), audio/Whisper transcription, MinIO storage, background upload-job tracking, and PDF report generation.
**Category:** RAG / LLM / Embeddings / Vector DB / OCR / STT/Audio / Document ingestion / File processing.
**Depends on:** `config.py`, `loaders/`, `utils/`.
**Depended on by:** `agent/`, `routes/`, `memory/` (via `llm_adapter.py`'s lazy import of `rag_service.get_llm()`).

### `backend/loaders/`
**Responsibility:** Per-file-type parsing (PDF, DOCX, TXT/MD, JSON, images, Excel/CSV) behind one dispatch function, `registry.load_document_from_bytes`. PDF/image loaders call into `services/ocr_service.py` when content isn't extractable as text.
**Category:** Document ingestion / File processing / OCR (PDF/image branch only).
**Depends on:** `services/ocr_service.py` (pdf/image loaders only), `config.py`.
**Depended on by:** `services/rag_service.py` (the only caller of `loaders.registry`).

### `backend/memory/`
**Responsibility:** Per-conversation conversational memory — an in-RAM sliding message window (`short_memory.py`) plus a persisted, deduplicated, LLM-extracted fact store (`summary_memory.py` + `fact_extractor.py`), coordinated by `memory_manager.py`. `llm_adapter.py` decouples fact extraction from any specific LLM client.
**Category:** Memory.
**Depends on:** `config.py`; lazily on `services/rag_service.get_llm()` (via `llm_adapter.py`, to avoid a circular import).
**Depended on by:** `agent/agent.py` (owns one `MemoryManager` per conversation), indirectly `agent/tools/generate_tool.py` and `respond_tool.py` (via an injected callback).

### `backend/routes/`
**Responsibility:** FastAPI HTTP + WebSocket surface. `chat.py` (text + voice chat, reset), `ws.py` (streamed chat), `upload.py` (document ingestion + lifecycle), `ocr.py` (standalone handwritten OCR), `reports.py` (PDF report generation/download), `health.py`.
**Category:** API layer / WebSocket streaming.
**Depends on:** `agent/session.py`, `services/*`, `utils/timing.py`, `utils/file_validation.py`.
**Depended on by:** `main.py` (mounts every router).

### `backend/utils/`
**Responsibility:** Cross-cutting helpers: `device.py` (shared CUDA/CPU resolution for the embedding model + cross-encoder), `timing.py` (per-request latency profiler via `contextvars`), `file_validation.py` (magic-byte/content validation before any parsing).
**Category:** Utilities / Observability (timing.py) / File processing (file_validation.py).
**Depends on:** `config.py`.
**Depended on by:** `services/embeddings_provider.py`, `services/rag_service.py`, `routes/upload.py`, virtually every module for `timing`.

### `backend/scripts/`
**Responsibility:** Standalone, manually-run offline evaluation harnesses for handwritten OCR (CER/WER benchmarking, batching experiments, preprocessing diagnostics). **Confirmed not part of the runtime pipeline** — zero imports of these scripts exist in `routes/`, `services/`, `loaders/`, or `main.py`.
**Category:** Other (developer tooling).

### `backend/tests/`
**Responsibility:** Pytest suite covering chunking, document lifecycle (reindex/delete), file validation, and handwritten OCR correctness/batching.
**Category:** Other (test suite).

### `backend/memory_storage/`
**Responsibility:** Runtime data directory — one JSON file per conversation_id holding its persisted fact store. Not code; contains a real example file, `f10ca108-b192-4c49-815e-3e8bef4017eb.json`, whose schema was verified directly (see §15).
**Category:** Memory (persistence).

### `frontend/app/`, `frontend/components/`, `frontend/lib/`, `frontend/services/`
**Responsibility:** The chat UI, upload UI, voice recorder, handwritten-OCR modal, report download card, conversation-id management (sessionStorage-scoped), and the single module (`services/api.ts`) that talks to every backend endpoint (REST + WebSocket).
**Category:** API layer (client side) / WebSocket streaming (client side).
**Depends on:** Backend routes exclusively via `services/api.ts`.

---

## 4. File-by-File Explanation

> Grouped by subsystem. Every file below was read in full. "Called by" / "Calls" are grep-verified, not inferred.

### 4.1 Configuration & entrypoint

#### `backend/config.py`
**Purpose:** Single source of truth for every environment-driven setting (`Settings` class, instantiated once as `settings`). Every other module reads config through `from config import settings` — nothing reads `os.environ` directly elsewhere.
**Main components:** `Settings` class (~370 lines of grouped constants — LLM/Groq, embeddings, Qdrant, RAG/reranking, chunking, Excel ingestion, audio/OCR, handwritten OCR, agent, agent lifecycle, profiling/debugging, memory, MinIO, report generation); helpers `_bool`/`_int`/`_float` for typed env parsing.
**Inputs:** `.env` file (via `python-dotenv`), OS environment.
**Outputs:** The `settings` singleton object.
**Dependencies:** none internal (leaf module).
**External dependencies:** `python-dotenv`.
**Called by:** every other backend module.
**Status:** Core / actively used.

#### `backend/main.py`
**Purpose:** FastAPI app construction — CORS middleware (explicit allowlist via `FRONTEND_ORIGIN`, never `"*"` with credentials), mounts all six routers, and on startup calls `rag_service.load_existing_db()` to attach to any pre-existing Qdrant collection.
**Main components:** `app = FastAPI(...)`; `on_startup()` handler.
**Dependencies:** `config.py`, all of `routes/*`, `services/rag_service.load_existing_db`.
**External dependencies:** FastAPI, `uvicorn` (dev run only, `if __name__=="__main__"`).
**Called by:** `uvicorn main:app` (Docker CMD / dev server).
**Status:** Core / actively used.

### 4.2 Agent

#### `backend/agent/agent.py`
**Purpose:** The ReAct loop. One `Agent` instance = one conversation's planner + memory + active-document state.
**Main components:** `Agent` class — `run()`/`run_stream()` (public entry points), `_run_impl`/`_run_stream_impl` (the loop itself), `_build_messages`, `_run_tool`, `_correct_premature_terminal` (deterministic backstop, no extra LLM call — see §7), `_update_active_document_from_retrieval`, `_remember`, lifecycle helpers `touch`/`is_idle`/`_mark_busy` (used by `session.py`'s idle-eviction cleanup). Module-level `_looks_like_small_talk()` + `_SMALL_TALK_PHRASES` (bilingual greeting/farewell whitelist, ≤6 words, used only by the backstop).
**Inputs:** `question: str`, `language: "auto"|"ar"|"en"`.
**Outputs:** `ExecutionContext` (non-streaming) or a generator of `{"type": "token"|"status"|"done", ...}` dicts (streaming).
**Dependencies:** `memory.llm_adapter.LLMTextGenerator`, `memory.memory_manager.MemoryManager`, `services.rag_service` (`detect_language`, `build_sources_from_dicts`, plus streaming helpers), `agent.llm.AgentLLM`, `agent.prompt.{SYSTEM_PROMPT,USER_PROMPT}`, `agent.registry.build_tools`, `agent.schemas.*`, `utils.timing`.
**Called by:** `routes/chat.py::_run_agent`, `routes/ws.py::_stream_answer`.
**Runtime role:** The single orchestration point every chat request passes through, both HTTP and WebSocket.
**Important behavior:** Max 6 iterations (`AGENT_MAX_ITERATIONS`, default 6) before forcing a final answer; duplicate-retrieve guard (`context.retrieved_questions`); `_correct_premature_terminal` deterministically forces a `retrieve` action (no extra Groq round-trip) if the planner picks `respond`/`generate` on turn 1 with nothing retrieved yet and the message isn't small talk — this replaced an earlier design that re-asked the LLM a second time (removed specifically to cut Groq call volume, see the method's docstring); in-flight/idle tracking via `threading.Lock` + `_in_flight` counter so a busy Agent is never evicted mid-request.
**Status:** Core / actively used.

#### `backend/agent/llm.py`
**Purpose:** Wraps the Groq model used for the planner's structured JSON action output.
**Main components:** `AgentLLM` class — `invoke(messages, fallback_question)` retries up to `max_retries=2` on invalid JSON/schema mismatch (re-prompting with a correction message), then falls back to a deterministic `RetrieveAction` (`_fallback_action`) rather than crashing the turn.
**Dependencies:** `services.llm_provider.get_agent_llm`, `agent.schemas` (Pydantic `AgentAction` discriminated union, validated via `TypeAdapter`).
**External dependencies:** `pydantic`.
**Called by:** `agent/agent.py` (`self.llm.invoke(...)` in both `_run_impl` and `_run_stream_impl`).
**Status:** Core / actively used.

#### `backend/agent/prompt.py`
**Purpose:** The planner's system + user prompt templates. The agent LLM **never writes the final answer** here — only picks the next tool.
**Main components:** `SYSTEM_PROMPT` (tool catalogue, semantic-intent-recognition rules for report/compare, the coreference-resolution "HARD RULE" forcing `retrieve` before any clarifying question, multi-question handling, spreadsheet-aware `top_k` guidance, strict JSON output format), `USER_PROMPT` (templated with question, active document, memory, retrieved-doc count, observations, previously-retrieved questions).
**Dependencies:** none (pure string templates).
**Called by:** `agent/agent.py::_build_messages`.
**Status:** Core / actively used.

#### `backend/agent/registry.py`
**Purpose:** Factory building the per-conversation tool dict (not module-level singletons, since `generate`/`respond` need an injected `memory_text_provider` callback).
**Main components:** `build_tools(memory_text_provider, conversation_id, active_document_provider, active_document_setter) -> dict[str, Tool]`.
**Dependencies:** all six `agent/tools/*.py` classes.
**Called by:** `agent/agent.py::__init__`.
**Status:** Core / actively used.

#### `backend/agent/schemas.py`
**Purpose:** Pydantic contracts for the planner's action space and the state threaded through the loop.
**Main components:** `ToolName` enum (`retrieve, generate, summarize, compare, respond, report`); `TERMINAL_TOOLS` frozenset; per-tool `*Arguments` models; per-tool `*Action` models; `AgentAction` (discriminated union on `action` field); `ExecutionContext` (`documents`, `observations`, `retrieved_questions`, `summary`/`comparison`/`answer`, `report`, `needs_clarification`, `language`, `final_answer()` helper returning whichever terminal output was produced).
**Called by:** every file in `agent/` and `agent/tools/`.
**Status:** Core / actively used.

#### `backend/agent/session.py`
**Purpose:** Process-wide `conversation_id -> Agent` registry with idle-eviction so memory doesn't leak unboundedly.
**Main components:** `get_agent(conversation_id)` (creates-or-returns, calls `agent.touch()`), `reset_agent(conversation_id)`, `_cleanup_once()`/`_cleanup_loop()` (daemon thread, runs every `AGENT_CLEANUP_INTERVAL_SECONDS`=300s, evicts entries idle ≥ `AGENT_IDLE_TIMEOUT_SECONDS`=1800s **and** with zero in-flight requests).
**Important behavior:** Eviction only frees RAM — `memory_storage/<id>.json` on disk is untouched; the next request for the same `conversation_id` gets a fresh `Agent` whose long-term facts are reloaded from disk (short-term memory starts empty). Cleanup thread is a **module-level singleton started at import time** (`_cleanup_thread.start()` at module bottom).
**Called by:** `routes/chat.py`, `routes/ws.py`.
**Status:** Core / actively used.

#### `backend/agent/tools/retrieve_tool.py`
**Purpose:** Non-terminal tool wrapping `rag_service.retrieve()`.
**Main components:** `RetrieveTool.run(context, question, top_k=5, raw_question="")` — dedupes newly retrieved chunks against `context.documents` by id, appends an observation dict (`chunks_added`, `total_documents`, `sources`).
**Important behavior:** Wraps the call in `timing.substage("retrieval_total_ms")` (not `stage()`, deliberately, to avoid double-counting wall-clock inside `rag_service.retrieve`'s own internal stages).
**Status:** Core / actively used.

#### `backend/agent/tools/generate_tool.py`
**Purpose:** Terminal tool — final answer from retrieved documents + memory.
**Main components:** `GenerateTool.run(context, question)` — falls back to `rag_service.answer_from_memory` if no documents were retrieved but memory text exists; else calls `rag_service.generate_answer`.
**Status:** Core / actively used.

#### `backend/agent/tools/summarize_tool.py` / `compare_tool.py` / `respond_tool.py`
**Purpose:** Terminal tools mirroring `generate_tool.py`'s shape for summarize/compare/respond-from-memory-only.
**Status:** Core / actively used (all three).

#### `backend/agent/tools/report_tool.py`
**Purpose:** Terminal tool generating a PDF report — either whole-document (`run()`) or topic-scoped (`_run_topic_report()`).
**Main components:** Target-document resolution order: explicit filename → conversation's active document → the only uploaded document → ask the user (`context.needs_clarification`). `_fuzzy_match()` (exact/substring, then `difflib.get_close_matches`, cutoff 0.6).
**Dependencies:** `services.rag_service` (`list_stored_files`), `services.report_service` (`generate_report`, `generate_topic_report`).
**Important behavior:** The whole-document path is **not** scoped by `conversation_id` (reads a named file directly from the global registry) — a documented, deliberate limitation, distinct from the topic-scoped path which retrieves via the conversation-scoped vector search.
**Status:** Core / actively used.

### 4.3 RAG / LLM / Embeddings / Vector DB core

#### `backend/services/rag_service.py` (2,327 lines — the largest file in the AI layer)
**Purpose:** The RAG engine: ingestion, query-variant generation, lexical + cross-encoder reranking, MMR diversification, prompt construction, and the full agent-facing + direct (`/chat`) public API.
**Main components (selected):**
- Vector DB state: `_get_vector_db()`, `_refresh_retriever()`, `load_existing_db()`, `is_ready()`.
- Document isolation: `_conversation_filter()`, `_document_filter()` (Qdrant metadata filters — see §11).
- Query processing: `detect_language()`, `_normalize_arabic()`, `_normalize()`, `_keywords()`, `_ngrams()`, `_translate()` (LLM, `lru_cache(512)`), `_rewrite_query()` (LLM, combined typo-fix+synonym-expand, `lru_cache(256)`), `_query_variants()` (builds up to 22 retrieval variants; runs rewrite+translate **concurrently** via `_run_concurrent`), `_add_raw_question_anchor()` (deterministically anchors the user's literal wording alongside the planner's possibly-reformulated query, since LLM output isn't guaranteed stable run-to-run even at temperature 0).
- Reranking: `_lex_score()` (unigram+bigram overlap), `_get_cross_encoder()` (lazy singleton, permanent fallback to lexical-only on load failure), `_rerank()` (blends cross-encoder + lexical, `alpha=RERANK_ALPHA`), `_diversify()` (greedy MMR-lite).
- Core retrieval: `_retrieve()` — embeds all variants in **one batched call**, fans out Qdrant searches concurrently, dedupes, applies `CONFIDENCE_THRESHOLD` as a coarse near-zero-overlap filter, widens `top_n` for Excel-sourced candidates.
- Prompt construction: `build_prompt()` (bilingual, strict grounding rules — see §9 excerpt), `build_prompt_with_memory()`, `_clean_answer()`, `_build_sources()`/`build_sources_from_dicts()`.
- Ingestion: `update_db_files()` (fresh upload → Qdrant), `reindex_document()` (prepare-new-before-delete-old ordering), `_chunk_documents()` (dispatches to recursive/semantic/hybrid strategies), `_semantic_split_documents()`, `_hybrid_split_documents()`.
- Agent-facing public API: `retrieve()`, `generate_answer()`/`generate_answer_stream()`, `summarize()`/`summarize_stream()`, `compare()`/`compare_stream()`, `answer_from_memory()`/`answer_from_memory_stream()`.
- File registry: `_load_registry()`/`_save_registry()` (JSON at `PROCESSED_FILES_REGISTRY`), `list_stored_files()`, `delete_conversation_documents()`, `delete_document()`.
**Dependencies:** `loaders.registry`, `services.db_service`, `services.embeddings_provider`, `services.llm_provider`, `services.storage_service`, `utils.timing`, `config`.
**External dependencies:** `langchain_core`, `langchain_qdrant`, `langchain_text_splitters`, `qdrant_client`, `numpy`.
**Called by:** `agent/tools/*.py`, `agent/agent.py`, `routes/chat.py`, `routes/upload.py`, `routes/ocr.py`, `services/report_service.py`.
**Status:** Core / actively used.

#### `backend/services/llm_provider.py`
**Purpose:** The **only** module that imports the `groq` SDK — every LLM call in the app funnels through here.
**Main components:** `GroqLLM` class — `invoke(prompt)` (single-prompt), `chat(messages, json_mode=False)` (role-tagged, used by the planner), `stream(prompt)`/`stream_chat(messages)` (token generators). `get_llm()` (shared singleton, model=`GROQ_MODEL`) and `get_agent_llm(model=None)` (shared singleton, model=`AGENT_MODEL`, or a fresh instance if a model override is passed). `_log_outgoing_messages()` — debug-only (`AGENT_DEBUG` gated), logs the exact wire request for every Groq call across the whole app (planner, query rewrite/translate, generate/respond/summarize/compare, memory fact-extractor, report map/reduce).
**Dependencies:** `config`.
**External dependencies:** `groq`.
**Called by:** `rag_service.py` (module-level `llm = _get_shared_llm()`), `agent/llm.py` (`get_agent_llm`), `report_service.py` (`get_llm`), `memory/llm_adapter.py` (indirectly via `rag_service.get_llm()`).
**Status:** Core / actively used.

#### `backend/services/embeddings_provider.py`
**Purpose:** Local, on-device embedding model — no API, no key.
**Main components:** `LocalEmbeddings(Embeddings)` — `_encode()` (wraps `SentenceTransformer.encode`, `normalize_embeddings=True`), `embed_documents()` (prefixes `"passage: "`), `embed_query()` (prefixes `"query: "`), `embed_queries()` (batched multi-query embedding — explicitly chosen over N concurrent single calls because concurrent small GPU calls measured ~10x slower than one batched call, per `PROFILING.md`). `get_embeddings()` — module-level singleton, loaded once.
**Configuration:** `EMBEDDING_MODEL` (default `intfloat/multilingual-e5-large`), `EMBEDDING_DEVICE` (`auto`/`cpu`/`cuda`, via `utils.device.resolve_device()`), `EMBEDDING_PROVIDER` (only `"local"` supported — anything else logs a warning and falls back).
**Called by:** `rag_service.py` (module-level `embeddings = get_embeddings()`), `db_service.ensure_collection()` (to size the Qdrant collection).
**Status:** Core / actively used.

#### `backend/services/db_service.py`
**Purpose:** Thin Qdrant client wrapper — server mode only (`QDRANT_URL`), never embedded/file mode.
**Main components:** `get_client()` (singleton), `with_retries()` (bounded retry/backoff wrapper used by every Qdrant call site), `ensure_collection()` (creates with `Distance.COSINE` if missing; **verifies** schema — raises `RuntimeError` rather than silently recreating — if it already exists with a mismatched vector size/distance), `is_available()` (health-check probe, no retries).
**Configuration:** `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`, `QDRANT_TIMEOUT_SECONDS`, `QDRANT_CONNECT_RETRIES` (5), `QDRANT_RETRY_DELAY_SECONDS` (2.0).
**Called by:** `rag_service.py`, `routes/health.py`.
**Status:** Core / actively used.

#### `backend/services/report_service.py`
**Purpose:** Map-reduce PDF report generation (whole-document or topic-scoped), rendered with `reportlab`, Arabic-shaped via `arabic_reshaper`/`python-bidi`.
**Main components:** `_map_extract()` (per-slice structured JSON extraction, one LLM call per ~`REPORT_MAP_CHUNK_CHARS`-sized slice, run on a bounded `ThreadPoolExecutor(MAP_EXTRACT_CONCURRENCY=5)`), `_aggregate()`/`_dedupe_with_pages()`, `_reduce_narrative()` (4 concurrent LLM calls: executive summary, introduction, relationships, conclusion), `build_report_data()`, `_topic_is_covered()` (LLM relevance gate for topic reports — the real "not enough info" guard, since retrieval always returns *something*), `build_topic_report_data()`, `render_report_pdf()` (two-pass `multiBuild` for a real, page-numbered table of contents), `generate_report()`/`generate_topic_report()` (public entrypoints, upload PDF to MinIO).
**Dependencies:** `services.rag_service` (`detect_language`, `get_document_pages`, `retrieve`), `services.storage_service`, `services.llm_provider.get_llm`.
**External dependencies:** `reportlab`, `arabic_reshaper`, `python-bidi`.
**Called by:** `agent/tools/report_tool.py`, `routes/reports.py`.
**Status:** Core / actively used.

### 4.4 Document ingestion & loaders

#### `backend/routes/upload.py`
**Purpose:** Upload/lifecycle HTTP surface — streaming size-bounded upload to temp disk, content validation, background ingestion job kickoff, job-status polling, stored-file listing/download, reindex, delete.
**Main components:** `_stream_upload_to_temp_file()` (1MB chunks, `MAX_UPLOAD_SIZE_MB` enforced incrementally), `_validate_temp_file()` (→ `utils.file_validation.validate_upload`), `_ingest_job()` (background-thread target calling `rag_service.update_db_files`), routes `POST /upload`, `GET /upload/status/{job_id}`, `GET /stored-files`, `GET /files/{object_name}/download`, `POST /documents/{id}/reindex`, `DELETE /documents/{id}`.
**Important behavior:** Validation is synchronous in the request; only parse→chunk→embed→index is backgrounded via `asyncio.create_task(asyncio.to_thread(_ingest_job, ...))`, returning `{"job_id","status":"queued"}` immediately. Delete is the one lifecycle op that stays synchronous (`await asyncio.to_thread(delete_document, ...)`, no job/polling) since it's fast.
**Status:** Core / actively used.

#### `backend/services/upload_jobs.py`
**Purpose:** In-process, in-memory job registry (`job_id -> {status, stage, chunks_added, error, ...}`) so `POST /upload` can return instantly and the frontend can poll. **Not persistent** — lost on backend restart.
**Main components:** `create_job()`, `set_stage()`, `mark_done()`, `mark_error()`, `get_job()` (prunes entries older than `_JOB_TTL_SECONDS`=3600 on lookup).
**Called by:** `routes/upload.py`.
**Status:** Core / actively used.

#### `backend/services/storage_service.py`
**Purpose:** MinIO wrapper for original uploads (`MINIO_BUCKET_UPLOADS`) and generated reports (`MINIO_BUCKET_REPORTS`). Optional dependency — every function raises `StorageUnavailableError` (never crashes the process) if the `minio` package is missing or the server is unreachable.
**Main components:** `get_client()` (internal-endpoint singleton), `_get_public_client()` (separate singleton for browser-facing presigned URLs, since `MINIO_ENDPOINT` may be a Docker-internal hostname), `upload_bytes()`, `download_bytes()`, `presigned_url()` (returns `None`, not an exception, on failure — callers fall back to the backend-proxied download route).
**Status:** Core / actively used.

#### `backend/loaders/registry.py`
**Purpose:** The single file-type-detection/dispatch point.
**Main components:** `_EXT_TO_TYPE` (pdf/docx/doc/txt/md/json/6 image extensions/xlsx/xls/csv), `SUPPORTED_EXTENSIONS`, `_DISPATCH` (ext → loader function), `get_file_type()`, `get_content_type()`, `load_document_from_bytes()`.
**Called by:** `rag_service._load_document_from_bytes` (only caller), `utils/file_validation.py`.
**Status:** Core / actively used.

#### `backend/loaders/base.py`
**Purpose:** Shared helpers — `make_meta(filename, file_type, **extra)` (builds the standard metadata dict: `source`, `file_type`, `page`, `timestamp`), `clean_text()`.
**Status:** Core / actively used.

#### `backend/loaders/pdf_loader.py`
**Purpose:** PDF loading with embedded-text extraction + OCR fallback decision logic (see §10).
**Main components:** `load(filename, data)` — the whole module. Decides whole-document-scanned (`len(text) < OCR_MIN_TEXT_CHARS` → OCR entire PDF as one blob) vs. mixed-document (per-page weak-text detection → OCR only those pages).
**Dependencies:** `services.ocr_service` (`perform_ocr_pdf_bytes`, `perform_ocr_pdf_pages_bytes`).
**Status:** Core / actively used.

#### `backend/loaders/docx_loader.py`
**Purpose:** DOCX/DOC via `Docx2txtLoader` (LangChain wrapper over `docx2txt`), requires a temp file (loader takes a path, not bytes).
**Status:** Core / actively used.

#### `backend/loaders/text_loader.py`
**Purpose:** `load_text()` (txt/md, UTF-8 decode with `errors="replace"`), `load_json()` (parse+pretty-reserialize, falls back to raw text on parse failure).
**Status:** Core / actively used.

#### `backend/loaders/excel_loader.py`
**Purpose:** Excel/CSV → **final, pre-sized** chunks (sheet summaries + overlapping row groups), deliberately bypassing `rag_service`'s text splitter.
**Main components:** `_read_sheets()` (pandas, `openpyxl` for xlsx, `xlrd` for legacy xls), `_rows_per_group()` (derives group size from `CHUNK_SIZE`/`CHUNK_OVERLAP` + sampled average row length), `_build_summary_chunk()`, `_build_row_group_chunks()` (sliding window with overlap), `load()`.
**Status:** Core / actively used.

#### `backend/loaders/image_loader.py`
**Purpose:** 16-line wrapper — `perform_ocr_image_bytes()` → `Document` if non-empty, else `[]`. All strategy/PSM logic lives in `ocr_service.py`.
**Status:** Core / actively used.

#### `backend/utils/file_validation.py`
**Purpose:** Pre-parse content validation — extension allowlist + magic-byte/structural checks (PDF `%PDF-`, DOCX/XLSX ZIP magic, legacy DOC/XLS OLE magic, image via `PIL.Image.verify()`, text via a NUL-byte-in-first-4KB binary heuristic) — runs **before** any loader/OCR code.
**Status:** Core / actively used.

### 4.5 OCR (printed text)

#### `backend/services/ocr_service.py`
**Purpose:** Tesseract-based OCR with a tiered preprocessing/PSM-mode strategy sweep and an early-exit confidence check.
**Main components:** `OCR_STRATEGIES`/`OCR_PSM_MODES` (images: 5 strategies × 3 PSM modes), `PDF_OCR_STRATEGIES`/`PDF_OCR_PSM_MODES` (PDF pages: cheaper, 3×2), `_preprocess_for_ocr()` (adaptive/otsu/denoise/sharpen/contrast via OpenCV, upscales small images first), `_run_tesseract()` (`--oem 1 --psm N -l ara+eng`), `_ocr_result_confident()` (length ≥`OCR_MIN_TEXT_CHARS` **and** alnum-ratio ≥`OCR_MIN_ALNUM_RATIO`), `_ocr_image_tiered()` (tries the cheapest combo first, accepts immediately if confident, else runs the full remaining sweep and merges via `_merge_ocr_results` — dedup by normalized line, longest-source-first), `perform_ocr_image_bytes()`, `perform_ocr_pdf_bytes()` (whole-PDF, page-parallel via bounded `ThreadPoolExecutor`), `perform_ocr_pdf_pages_bytes()` (OCRs only specified page indices — used for mixed PDFs), `extract_text()` (a dispatch convenience function with **no verified callers** — see §20).
**Configuration:** `TESSERACT_CMD`, `OCR_MIN_TEXT_CHARS` (20), `OCR_MIN_ALNUM_RATIO` (0.6), `OCR_MAX_CONCURRENT_PAGES` (4).
**Called by:** `loaders/pdf_loader.py`, `loaders/image_loader.py`.
**Status:** Core / actively used (all functions except `extract_text`, which is Dead/unclear — see §20).

#### `backend/services/handwritten_ocr_service.py`
**Purpose:** Independent handwriting-recognition service (Hugging Face TrOCR), including its own from-scratch line-segmentation (classic OpenCV, no ML). Reachable **only** via `routes/ocr.py`.
**Main components:** `HandwrittenOCRService` class — `_resolve_device()` (its **own** CUDA/CPU auto-detect, independent of `utils/device.py`, no manual-override setting exists for it), `_get_model()` (double-checked-locking lazy load, cached forever per language), `recognize()`/`recognize_with_debug()`, `_segment_lines()` (gradient-based ink mask + adaptive row threshold + band merge/split + aspect-ratio-aware crop widening), `_recognize_lines()` (sequential for ≤1 line, batched in groups of `HANDWRITTEN_OCR_MAX_BATCH_SIZE` otherwise, with per-sub-batch fallback to sequential on error). `get_handwritten_ocr_service()` — module singleton.
**Configuration:** `HANDWRITTEN_OCR_EN_MODEL` (default `microsoft/trocr-small-handwritten`), `HANDWRITTEN_OCR_AR_MODEL` (default `RayR1/trocr-base-arabic-handwritten`), `HANDWRITTEN_OCR_MAX_NEW_TOKENS` (256), `HANDWRITTEN_OCR_MAX_BATCH_SIZE` (8).
**Called by:** `routes/ocr.py` only (plus `tests/test_handwritten_ocr.py` and the offline `scripts/evaluate_*.py`).
**Status:** Supporting / actively used — real feature, but not wired into automatic ingestion (confirmed: zero references in `loaders/`, `rag_service.py`, or `routes/upload.py`).

#### `backend/routes/ocr.py`
**Purpose:** `POST /api/ocr/handwritten` — the only entry point for TrOCR. Optional pipeline integration: if `index=true` **and** `conversation_id` are both supplied, the extracted text is fed into `rag_service.update_db_files` as a synthetic `.handwritten-ocr.txt` file, indexed exactly like any normal upload.
**Status:** Core / actively used.

### 4.6 Audio / STT

#### `backend/services/audio_service.py`
**Purpose:** Standalone speech-to-text — ffmpeg conversion, silence detection, Whisper transcription.
**Main components:** `_get_model()` (lazy singleton, `whisper.load_model(WHISPER_MODEL_NAME)` — **`openai-whisper`**, not `faster-whisper`), `_run_ffmpeg()` (subprocess, retries once with a bare `"ffmpeg"` PATH lookup if `FFMPEG_PATH` fails), `_get_audio_rms_db()` (ffmpeg `volumedetect` filter), `_convert_to_wav()` (16kHz mono PCM16), `_detect_language()` (Whisper encoder-only language-ID pass, Arabic favored via `ar >= en * 0.75`), `_initial_prompt()` (hardcoded, dialect-tuned Arabic prompt vs. a plain English one), `transcribe_audio()` (main entry point), `transcribe_audio_path()` (no verified callers anywhere in the repo — see §20).
**Configuration:** `WHISPER_MODEL_NAME` (default `"small"`), `SILENCE_THRESHOLD_DB` (default -60.0), `FFMPEG_PATH` (default `"ffmpeg"`).
**Important behavior:** `model.transcribe()` uses Whisper's full default multi-temperature retry schedule `(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)` plus `beam_size=5, best_of=5, condition_on_previous_text=False` — deliberately, per an inline comment, since a bare `temperature=0.0` would silently disable Whisper's internal quality-gate retries.
**Called by:** `routes/chat.py::chat_voice` only (via `asyncio.to_thread`).
**Status:** Core / actively used (`transcribe_audio`); `transcribe_audio_path` is Dead code / unclear (no call sites found).

### 4.7 Memory

#### `backend/memory/short_memory.py`
**Purpose:** Pure in-RAM message buffer, no I/O, no LLM.
**Main components:** `ShortMemory` — `add_message()`, `get_messages()`, `should_summarize()` (message-count **or** char-count threshold), `keep_last()`.
**Status:** Core / actively used.

#### `backend/memory/llm_adapter.py`
**Purpose:** Decouples fact extraction from a concrete LLM client.
**Main components:** `LLMTextGenerator.generate(prompt)` — lazily imports `services.rag_service.get_llm()` to avoid a circular import; swallows exceptions, returns `""`.
**Status:** Core / actively used.

#### `backend/memory/fact_extractor.py`
**Purpose:** LLM-based (not rule-based) extraction of atomic, categorized facts from a conversation window.
**Main components:** `build_prompt()` (existing facts + recent messages → instructs the model to keep facts atomic, merge near-duplicates, ignore small talk, emit a `remove` list for resolved facts, assign `category`∈{preference,decision,task,fact} and `importance` 1-5), `extract_facts()` (parses/validates the JSON response; merging/dedup is deliberately left to `FactStore`, not done here).
**Status:** Core / actively used.

#### `backend/memory/summary_memory.py`
**Purpose:** Long-term memory: capped, deduplicated fact store + its JSON persistence.
**Main components:** `FactStore` — `merge()` (fuzzy-match via `difflib.SequenceMatcher` at 0.92 similarity, replaces on match, drops on `remove_texts` match), `_cap()` (evicts lowest `(importance, updated_at)` once over `MEMORY_MAX_FACTS`), `render()` (recency+importance sorted, truncated to `MEMORY_SUMMARY_MAX_CHARS`). `SummaryMemory` — `load_facts()`/`save_facts()` (one JSON file per conversation at `MEMORY_STORAGE_DIR/<safe_id>.json`, schema `{"version":2,"conversation_id","facts":[...],"updated_at"}`; transparently upgrades legacy v1 `{"summary":...}` files into one synthetic fact), `delete_facts()`.
**Verified against the real file** `backend/memory_storage/f10ca108-b192-4c49-815e-3e8bef4017eb.json` — matches the v2 schema exactly.
**Status:** Core / actively used.

#### `backend/memory/memory_manager.py`
**Purpose:** The single class `Agent` talks to — coordinates short-term + long-term memory and triggers background fact extraction.
**Main components:** `MemoryManager` — `add_message()`/`add_turn()`, `as_prompt_text()` (renders `"Known facts...\n{render}\n\nRecent messages...\n{last MEMORY_WINDOW messages}"` — the single function everything downstream uses), `reset()`, `_summarize_async()` (synchronously trims short memory to `MEMORY_KEEP_RECENT`, then spawns a **daemon background thread** to run `extract_facts` + `FactStore.merge` + `SummaryMemory.save_facts`; failures logged, never propagated).
**Configuration:** `MEMORY_MAX_MESSAGES` (25), `MEMORY_KEEP_RECENT` (4), `MEMORY_WINDOW` (6), `MEMORY_MAX_CHARS` (12000), `MEMORY_MAX_FACTS` (40), `MEMORY_SUMMARY_MAX_CHARS` (1200), `MEMORY_STORAGE_DIR`.
**Called by:** `agent/agent.py` (owns one instance per conversation), `agent/tools/generate_tool.py`/`respond_tool.py` (via injected `memory_text_provider`).
**Status:** Core / actively used.

### 4.8 Routes

#### `backend/routes/chat.py`
**Purpose:** HTTP chat surface — `POST /chat` (text), `POST /chat/voice` (audio→text→same agent path), `POST /chat/reset`.
**Important behavior:** `_run_agent()` is defined once and called by **both** `/chat` and `/chat/voice` — voice queries converge on the exact same `agent.run()` call as typed queries, no separate voice code path. Both LLM-bearing calls (`transcribe_audio`, `_run_agent`) are offloaded via `asyncio.to_thread` so a slow request doesn't stall the event loop.
**Status:** Core / actively used.

#### `backend/routes/ws.py`
**Purpose:** `WS /ws/chat` — streams the agent's terminal answer token-by-token. Contains **no audio handling** (verified: zero matches for audio/voice/mic in this file).
**Main components:** `ws_chat()` (receive loop, JSON text frames both directions), `_stream_answer()` (runs `agent.run_stream()` on a worker thread via `asyncio.to_thread`, relays yielded events through an `asyncio.Queue` back to the socket).
**Important behavior:** The producer task is created (`asyncio.create_task`) **before** the `to_thread` hop specifically so `contextvars` (and therefore `utils.timing`'s per-request profiler) survive the thread hop — documented inline.
**Status:** Core / actively used.

#### `backend/routes/upload.py`, `backend/routes/ocr.py`
See §4.4/§4.5 above.

#### `backend/routes/reports.py`
**Purpose:** `POST /reports/generate` (whole-doc or topic-scoped), `GET /reports/{object_name}/download` (backend-proxied MinIO stream, fallback for when a direct presigned URL isn't reachable).
**Status:** Core / actively used.

#### `backend/routes/health.py`
**Purpose:** `GET /health` — reports MinIO + Qdrant reachability.
**Status:** Core / actively used.

### 4.9 Utilities

#### `backend/utils/device.py`
**Purpose:** Shared CUDA/CPU resolution for the embedding model and cross-encoder reranker (only — **not** used by `handwritten_ocr_service.py`, which has its own independent auto-detect).
**Main components:** `resolve_device()` — reads `EMBEDDING_DEVICE` (`auto`/`cpu`/`cuda`), caches the decision process-wide.
**Status:** Core / actively used.

#### `backend/utils/timing.py`
**Purpose:** Per-request latency profiler using `contextvars` so the timer is visible across every module a request touches without threading a parameter through every call.
**Main components:** `RequestTimer` (`record_stage`, `record_substage`, `mark`, `report`), `start()`/`stage()`/`substage()`/`finish()` module-level API, `run_concurrent_ctx()` (propagates `contextvars.copy_context()` into a `ThreadPoolExecutor` so worker-thread `stage()` calls don't silently no-op).
**Configuration:** `LOG_REQUEST_PROFILE` (bool, default True).
**Status:** Core / actively used.

#### `backend/utils/file_validation.py`
See §4.4.

### 4.10 Offline / developer tooling (not runtime)

`backend/scripts/evaluate_handwritten_ocr.py`, `evaluate_ocr_followup.py`, `evaluate_ocr_preprocessing_check.py` — CER/WER benchmarking and preprocessing diagnostics for TrOCR, run manually (`python scripts/...`), reaching into `handwritten_ocr_service.py`'s private methods for benchmarking. **Status: Experimental / developer tooling, confirmed not imported anywhere in the runtime path.**

`backend/tests/*.py` — pytest suite (chunking, document lifecycle, file validation, handwritten OCR). **Status:** Supporting (quality assurance, not runtime).

### 4.11 Frontend

#### `frontend/services/api.ts`
**Purpose:** The single module centralizing all backend I/O.
**Main components:** `apiFetch()` (fetch wrapper, `AbortController` timeout, FastAPI `{"detail":...}` error parsing), `askQuestion()` (`POST /api/chat` — defined but **not called** from `ChatBox.tsx`, which uses the WebSocket instead; unclear/possibly-unused, see §20), `askVoice()` (`POST /api/chat/voice`, multipart), `resetConversation()`, `uploadFiles()`/`startUploadJob()`/`getUploadJobStatus()` (poll every `UPLOAD_POLL_INTERVAL_MS`=1000ms up to 15 min), `getStoredFiles()`, `ocrHandwritten()` (`POST /api/ocr/handwritten`, 5-minute timeout for first-run model download), `generateReport()` (`POST /api/reports/generate` — exported, **no caller found**, see §20), `streamChat()` (the WebSocket client — see §14), `wsUrl()` (resolves `NEXT_PUBLIC_WS_URL` or defaults to `ws://<host>:8000/ws/chat`).
**Status:** Core / actively used (most exports); `askQuestion`/`generateReport` unclear/likely unused.

#### `frontend/components/ChatBox.tsx`
**Purpose:** Core chat UI and streaming orchestrator.
**Main components:** `Message` type, `handleSubmit()` (text path → `streamChat`), `handleVoice()` (voice path → `askVoice`), `addUserMessage`/`appendAIToken`/`finalizeAIMessage`/`failAIMessage`/`setAIStatus` (message-state reducers), reset-on-`resetSignal` effect.
**Status:** Core / actively used.

#### `frontend/components/AnswerBox.tsx`, `SourceBox.tsx`, `ReportCard.tsx`
**Purpose:** Presentational — typing-dots/streaming-cursor rendering, `"Sources: A | B | C"` string parsed into pill badges (format matches `rag_service.py`'s `_build_sources`/`build_sources_from_dicts` exactly), report download card (`download_url` preferred, `proxy_download_path` fallback).
**Status:** Core / actively used (all three).

#### `frontend/components/VoiceRecorder.tsx`
**Purpose:** Browser mic capture via `MediaRecorder` (100ms timeslices), assembles a `Blob` tagged `audio/webm`, hands it to a parent-supplied `onResult` callback. Transmits nothing itself.
**Status:** Core / actively used.

#### `frontend/components/HandwrittenOcrModal.tsx`
**Purpose:** Standalone modal calling `ocrHandwritten()`; only reads `res.text` from the response (the `indexed`/`chunks_added` fields the backend can return are not surfaced in this UI).
**Status:** Core / actively used.

#### `frontend/components/UploadBox.tsx`
**Purpose:** Drag-and-drop/click upload UI, stage-labeled progress bar, retry-on-error, "Recent Documents" list with download links.
**Note:** its `accept` attribute lists `.pdf,.docx,.doc,.txt,.md,.json,.png,.jpg,.jpeg,.xlsx,.xls,.csv` — `.tiff`/`.bmp`/`.webp` are supported server-side but omitted from this filter (only affects the OS picker's default filter, not a hard block).
**Status:** Core / actively used.

#### `frontend/lib/conversation.ts`
**Purpose:** `getConversationId()` — `sessionStorage`-scoped (not `localStorage`, deliberately, per its own comment: one id per browser tab, cleared on tab close), generated via `crypto.randomUUID()` on first use.
**Status:** Core / actively used.

#### `frontend/lib/fileTypeMeta.ts`, `frontend/components/ui/*`
**Purpose:** Generic presentational helpers/primitives (badge/card/empty-state/skeleton), not chat-pipeline-specific themselves.
**Status:** Supporting / actively used.

---

## 5. Complete Runtime Architecture

```
USER (browser)
  │
  ├─ types a question ─────────────────────────────────────────────┐
  │                                                                  │
  ├─ records voice ──────────────┐                                  │
  │                               │                                  │
  ▼                               ▼                                  ▼
VoiceRecorder.tsx           ChatBox.handleVoice           ChatBox.handleSubmit
  │ MediaRecorder                │                                  │
  │ → Blob(audio/webm)           │                                  │
  └──────────────────────────────┤                                  │
                                  ▼                                  ▼
                        api.ts askVoice()                 api.ts streamChat()
                        POST /api/chat/voice               WS /ws/chat
                        (multipart/form-data)               (JSON frames)
                                  │                                  │
                                  ▼                                  ▼
                        routes/chat.py::chat_voice        routes/ws.py::ws_chat
                        1. transcribe_audio() (Whisper)     │
                        2. _run_agent(stt_text, ...) ───┐   │
                                                          │   │
                                                          ▼   ▼
                                             agent/session.py::get_agent(conversation_id)
                                                          │
                                                          ▼
                                              agent/agent.py :: Agent.run() / run_stream()
                                                          │
                                       ┌──────────────────┼───────────────────┐
                                       ▼                  ▼                   ▼
                              agent/llm.py           agent/tools/*      memory/memory_manager.py
                              (planner: pick          (execute the         (read: as_prompt_text()
                               next action via         chosen action)       into planner + generate
                               Groq JSON mode)              │               prompts; write: add_turn()
                                                             ▼               after final answer)
                                              services/rag_service.py
                                       (retrieve / generate / summarize / compare /
                                        answer_from_memory [+ *_stream variants])
                                                             │
                                       ┌─────────────────────┼─────────────────────┐
                                       ▼                     ▼                     ▼
                        embeddings_provider.py         db_service.py         llm_provider.py
                        (query variant embedding,      (Qdrant client,        (Groq: translate,
                         MMR vectors, local              conversation-        rewrite, generate,
                         sentence-transformers)          scoped search)        summarize, compare)
                                                             │
                                                             ▼
                                                     Qdrant (vector DB)
                                                             │
                                                     ┌───────┴────────┐
                                                     ▼                ▼
                                          cross-encoder reranker   MMR-lite diversification
                                          (sentence_transformers   (_diversify, greedy
                                           CrossEncoder, lazy       relevance/similarity
                                           singleton, lexical        tradeoff)
                                           fallback on load failure)
                                                             │
                                                             ▼
                                                  build_prompt_with_memory()
                                                             │
                                                             ▼
                                                     llm.invoke() / llm.stream()
                                                        (Groq chat completion)
                                                             │
                              ┌──────────────────────────────┴──────────────────────────────┐
                              ▼ (HTTP /chat, /chat/voice)                                    ▼ (WS /ws/chat)
                    JSON {answer, sources, stt_text, report}                token-by-token {"type":"token",...}
                              │                                              then {"type":"done", answer, sources, report}
                              ▼                                                              │
                    ChatBox.resolveAIMessage()                              ChatBox.appendAIToken() → finalizeAIMessage()
                              │                                                              │
                              └──────────────────────────┬───────────────────────────────────┘
                                                          ▼
                                         AnswerBox (text) + SourceBox (citations)
                                              + ReportCard (if a report was generated)
```

---

## 6. Chat Pipeline

Two parallel entry points, **both funneling into the identical `Agent` object per `conversation_id`**:

1. **HTTP, non-streaming** — `POST /api/chat` (text) or `POST /api/chat/voice` (audio) → `routes/chat.py::_run_agent()` → `agent.run()` → one JSON response with the full answer.
2. **WebSocket, streaming** — `WS /ws/chat` → `routes/ws.py::_stream_answer()` → `agent.run_stream()` on a worker thread, relayed via `asyncio.Queue` → a sequence of `{"type":"token"|"status"|"start"|"done"|"error"}` frames.

Both routes call `agent.session.get_agent(conversation_id)` first — this is where a conversation's `Agent` (and its `MemoryManager`) is created-or-reused, and where `Agent.touch()` records activity for the idle-eviction cleanup thread.

Voice specifically: `chat_voice()` calls `transcribe_audio()` (Whisper, via `asyncio.to_thread`) **then** passes the resulting text string into the exact same `_run_agent()` helper the plain-text route uses — there is no "voice-aware" branch anywhere past transcription. Voice never goes through the WebSocket (`ws.py` has zero audio-related code, verified by grep).

---

## 7. Agent Architecture

**Class:** `agent/agent.py::Agent`, one instance per `conversation_id`, held in `agent/session.py`'s process-wide registry.

**Loop** (`_run_impl`/`_run_stream_impl`, up to `AGENT_MAX_ITERATIONS`=6 iterations):

```
User Question
      │
      ▼
_build_messages()  ── injects: question, active_document, memory.as_prompt_text(),
      │                 documents-so-far count, observations, previously-retrieved questions
      ▼
AgentLLM.invoke()  ── Groq JSON-mode call (model = AGENT_MODEL, default llama-3.1-8b-instant)
      │                 retries up to 2x on invalid JSON/schema, then deterministic fallback
      ▼
_correct_premature_terminal()  ── no-LLM-call backstop: if the planner picked
      │                             respond/generate on turn 1 with nothing retrieved
      │                             and the message isn't small talk, force "retrieve"
      ▼
  action.action ?
      │
      ├── "retrieve" ──► RetrieveTool.run() ──► rag_service.retrieve() ──► loop again
      │                   (dedupes against context.documents; skips if question
      │                    already retrieved this turn)
      │
      └── terminal (generate/summarize/compare/respond/report)
              │
              ├── non-streaming: tool.run() → full string synchronously
              │
              └── streaming: _stream_terminal_action() yields token-by-token
                  from the matching rag_service *_stream function
                  (report is the one exception: no token stream, emits a
                   {"type":"status"} event instead and runs to completion
                   on the same worker thread)
                              │
                              ▼
                  _remember() ── memory_manager.add_turn(question, final_answer)
                              │
                              ▼
                  ExecutionContext.final_answer() + build_sources_from_dicts()
                              │
                              ▼
                        Response to caller
```

If `AGENT_MAX_ITERATIONS` is exhausted without a terminal action, the agent forces a final answer from whatever context it has (`self.tools["generate"].run(...)` or the streaming equivalent) rather than erroring.

**Tools** (`agent/tools/`): `retrieve` (non-terminal, repeatable), `generate`/`summarize`/`compare`/`respond`/`report` (all terminal — `TERMINAL_TOOLS` frozenset in `schemas.py`).

**State:** `ExecutionContext` (Pydantic model) threaded through every step — accumulates `documents`, `observations`, `retrieved_questions`, and the eventual `answer`/`summary`/`comparison`/`report`.

**Error handling/retries:** Planner JSON failures retry in-process (max 2) then fall back deterministically to a `retrieve` action rather than raising. RAG/LLM-layer exceptions inside tools are caught by `rag_service.py`'s own functions (return an error string) rather than propagating — a chat request essentially never 500s from an LLM-layer failure; it degrades to an error message inside the answer text. The one exception: routes catch true exceptions (e.g. Qdrant/Groq client construction failures) and return HTTP 500/`{"type":"error"}`.

**Memory interaction:** read at every planning step (`_build_messages`) and inside `generate`/`respond` tool execution (via injected `memory_text_provider`); written once per completed turn (`_remember`), only if a final answer exists.

---

## 8. RAG Pipeline

| Stage | File / Function | Input | Output | Config |
|---|---|---|---|---|
| 1. Query input | `agent/tools/retrieve_tool.py::run` | planner's `question` (+ `raw_question` on first retrieve of a turn) | — | — |
| 2. Language detection | `rag_service.detect_language` | text | `"ar"`/`"en"` (Arabic-char vs Latin-char count) | — |
| 3. Query normalization | `rag_service._normalize`/`_normalize_arabic` | text | normalized string (diacritics stripped, letter unification) | — |
| 4. Query rewriting | `rag_service._rewrite_query` (LLM, cached) | query, lang | `(corrected, alternatives[≤3])` | `QUERY_EXPANSION_ENABLED` |
| 5. Translation | `rag_service._translate` (LLM, cached) | query, target_lang | translated string | — |
| 6. Query variants | `rag_service._query_variants` + `_add_raw_question_anchor` | question, lang | up to 22 deduped variant strings | — |
| 7. Embedding | `embeddings_provider.LocalEmbeddings.embed_queries` | variant list | one batched forward pass → vectors | `EMBEDDING_MODEL`, `EMBEDDING_DEVICE` |
| 8. Qdrant search | `rag_service._retrieve::_search_by_vector` (concurrent, per-vector) | precomputed vector | `similarity_search_with_score_by_vector`, filtered by `conversation_id` | `RETRIEVER_K` |
| 9. Candidate collection | `rag_service._retrieve` | per-variant hits | deduped `Document` list | — |
| 10. Cross-encoder reranking | `rag_service._rerank` / `_get_cross_encoder` | question + candidates | blended score (`alpha*CE + (1-alpha)*lexical`) | `RERANK_USE_CROSS_ENCODER`, `CROSS_ENCODER_MODEL`, `RERANK_ALPHA` |
| 11. Confidence scoring | `rag_service._retrieve` (top_score check) | reranked scores | reject-all if `top_score < CONFIDENCE_THRESHOLD` | `CONFIDENCE_THRESHOLD` |
| 12. MMR/diversification | `rag_service._diversify` | scored pool | top-N re-selected for relevance-vs-redundancy | `RERANK_DIVERSIFY`, `MMR_LAMBDA` |
| 13. Context construction | `rag_service._build_context` / `_chunk_label` / `_trim_to_budget` | ranked docs | labeled, char-budgeted context string | `MAX_CONTEXT_CHARS` |
| 14. Final generation | `rag_service.generate_answer`/`_stream` → `build_prompt_with_memory` → `llm.invoke`/`.stream` | context, question, memory | answer string / token stream | `GROQ_MODEL`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_TOP_P` |

**Confidence filter caveat (explicit in code comments):** `CONFIDENCE_THRESHOLD` is a coarse, lexical-overlap-only heuristic that only reliably rejects *near-zero-overlap* queries — it is **not** trusted as the real topic-relevance guard. The actual grounding guarantee is the LLM prompt rule in `build_prompt()`: "answer only if the context specifically covers the question, else say the information isn't available."

---

## 9. Document Ingestion

```
Frontend file select (UploadBox.tsx)
      │
      ▼
POST /api/upload  (multipart: files[], conversation_id)
      │
      ▼
routes/upload.py::upload_files
  1. _stream_upload_to_temp_file()  — 1MB chunks, MAX_UPLOAD_SIZE_MB enforced incrementally
  2. _validate_temp_file() → utils/file_validation.validate_upload()
       — extension allowlist + magic-byte/structural check (PDF/ZIP/OLE magic,
         PIL.Image.verify() for images, NUL-byte heuristic for text)
  3. upload_jobs.create_job() → returns {job_id, status:"queued"} IMMEDIATELY
  4. asyncio.create_task(asyncio.to_thread(_ingest_job, ...))  ── background from here on
      │
      ▼
_ingest_job()  (worker thread)
      │
      ▼
services/rag_service.py::update_db_files(files, conversation_id, on_progress)
  a. dedup check: sha256(bytes) scoped by (conversation_id, hash) against
     PROCESSED_FILES_REGISTRY (processed_files.json)               [stage: "parsing" reported]
  b. _save_uploaded_file() → MinIO (MINIO_BUCKET_UPLOADS)            (best-effort; ingestion
                                                                       proceeds even if MinIO down)
  c. _load_document_from_bytes() → loaders.registry.load_document_from_bytes()
       ext → loader:
         pdf            → pdf_loader.load          (embedded text + OCR fallback, see §10)
         docx, doc       → docx_loader.load          (Docx2txtLoader, needs temp file)
         txt, md         → text_loader.load_text     (UTF-8 decode)
         json            → text_loader.load_json     (parse + pretty reserialize)
         jpg/jpeg/png/    → image_loader.load         (always OCR — see §10)
         tiff/bmp/webp
         xlsx/xls/csv    → excel_loader.load          (pandas → pre-sized sheet_summary +
                                                          row_group chunks, see below)
  d. bilingual "enrich" pass (adds normalized-Arabic/lowercase-English blocks)
  e. _deduplicate() across files                                    [stage: "chunking" reported]
  f. _chunk_documents() — branches on CHUNKING_STRATEGY:
       "recursive" (default) → RecursiveCharacterTextSplitter(CHUNK_SIZE, CHUNK_OVERLAP)
       "semantic"             → _semantic_split_documents (per-sentence-window embedding
                                  breakpoints; falls back to recursive on failure/0 chunks)
       "hybrid"               → _hybrid_split_documents (cheap recursive base chunks +
                                  ONE batched embed call, merge similar adjacent chunks;
                                  falls back to recursive on failure/0 chunks)
       Excel-sourced docs bypass this splitter entirely (pre-chunked by excel_loader.py)
  g. ensure_collection(embeddings) + vdb.add_documents(chunks)       [stage: "embedding" reported]
  h. registry write-back (filename, MinIO key, file_type, chunk count,
     processed_at, conversation_id, document_id)                    [stage: "done" reported]
      │
      ▼
Frontend polls GET /api/upload/status/{job_id} every 1s (up to 15 min)
  → UploadBox shows STAGE_LABELS per stage → onUploaded() + refreshFiles()
```

**Excel specifics:** `_rows_per_group()` derives a row-count-per-chunk from `CHUNK_SIZE`/`CHUNK_OVERLAP` and a sampled average serialized-row length, bounded by `EXCEL_ROWS_PER_CHUNK_MIN`/`MAX` (3-50); one `sheet_summary` chunk per sheet (row/column counts, sample rows) plus overlapping `row_group` chunks; sheets over `EXCEL_MAX_ROWS_PER_SHEET` (20000) are truncated for indexing.

**PDF/scanned/mixed/image/OCR handling:** see §10.

**Reindex** (`services/rag_service.reindex_document`): same `_chunk_documents` pipeline, ordering deliberately "prepare & validate new → delete old → insert new" so a document is never left with zero retrievable content mid-operation, and old/new never coexist for more than the delete-then-insert window.

---

## 10. OCR Pipeline

Two **fully independent** OCR systems exist:

| | Printed text (automatic) | Handwritten (opt-in) |
|---|---|---|
| Engine | Tesseract (`pytesseract`) | TrOCR (Hugging Face `transformers`) |
| File | `services/ocr_service.py` | `services/handwritten_ocr_service.py` |
| Trigger | Automatic, inside the upload pipeline (`pdf_loader.py`, `image_loader.py`) | Explicit `POST /api/ocr/handwritten` call only |
| Languages | Arabic+English simultaneously (`-l ara+eng`) | One of `ar`/`en` per call, explicit (no auto-detect) |
| Device | CPU only (subprocess) | CPU/CUDA auto (own detection, independent of `utils/device.py`) |

### 10.1 Printed OCR flow (Tesseract)

**PDF → text (`loaders/pdf_loader.py::load`):**
1. `PyPDFLoader` extracts embedded text per page.
2. If whole-document text length `< OCR_MIN_TEXT_CHARS` (20) → treat as **entirely scanned**: `ocr_service.perform_ocr_pdf_bytes()` renders every page (`pdf2image.convert_from_bytes`, dpi=200) and OCRs each in a bounded `ThreadPoolExecutor` (`OCR_MAX_CONCURRENT_PAGES`=4 concurrent Tesseract subprocesses) → collapsed into **one** `Document`.
3. Else (mixed document): per-page weak-text detection (`len(page_text.strip()) < OCR_MIN_TEXT_CHARS`) → `perform_ocr_pdf_pages_bytes()` OCRs **only** those specific pages (cheap, page-targeted `pdf2image` render) → overwrites just those pages' content.

**Image → text (`loaders/image_loader.py::load`):** always calls `ocr_service.perform_ocr_image_bytes()` — no text-extraction step exists for images (they're always OCR'd).

**Tiered strategy/PSM sweep (`ocr_service._ocr_image_tiered`):**
1. Try the cheapest combo first: `strategies[0]` ("adaptive" preprocessing) + `psm_modes[0]` (PSM 6) — one Tesseract call.
2. `_ocr_result_confident(text)`: accept immediately if `len(text) >= OCR_MIN_TEXT_CHARS` (20) **and** `alnum_ratio >= OCR_MIN_ALNUM_RATIO` (0.6, Unicode-aware for Arabic+Latin).
3. If not confident, escalate: run every remaining `(strategy, psm)` combination (images: 5 strategies × 3 PSM = up to 14 more calls; PDF pages: 3×2 = up to 5 more), merge all non-empty results via `_merge_ocr_results` (dedup by normalized line, longest-source-first).
4. Preprocessing strategies (OpenCV): `adaptive` (Gaussian blur + adaptive threshold), `otsu`, `denoise` (`fastNlMeansDenoising` + Otsu), `sharpen` (unsharp mask + Otsu), `contrast` (CLAHE + Otsu). Small images (<800×600) are upscaled first.

**Concurrency:** page-level parallelism bounded by `OCR_MAX_CONCURRENT_PAGES` regardless of document size or CPU count; per-page failures are caught inside the worker and simply skip that page (one bad page never aborts the document); page order is preserved via index-based writes, not completion order.

**Failure handling:** every function in `ocr_service.py` catches internally and returns `""`/skips rather than raising; `perform_ocr_image_bytes` has one more last-resort fallback (a raw single Tesseract call) if the tiered sweep still returns nothing.

### 10.2 Handwritten OCR flow (TrOCR)

`POST /api/ocr/handwritten` (`routes/ocr.py`) → `asyncio.to_thread(_run_ocr)` → `HandwrittenOCRService.recognize()`:
1. `_preprocess()` — EXIF-transpose, RGB convert, resize only if too small/large (LANCZOS), autocontrast. **No binarization** (preserves thin strokes/Arabic diacritics).
2. `_segment_lines()` — classic CV (no ML): Scharr-gradient ink mask → per-image adaptive (Otsu-derived) row threshold → band merge/split (bounded by min/max height) → falls back to treating the whole image as one line if segmentation finds nothing or produces implausibly many (>80) bands.
3. `_recognize_lines()` — single line: direct model call. Multiple lines: batched in groups of `HANDWRITTEN_OCR_MAX_BATCH_SIZE` (8), with per-sub-batch fallback to sequential recognition on error.
4. Joined line texts → returned as `{"text","language","type":"handwritten"}`.

**Optional RAG integration:** only if the caller passes `index=true` **and** `conversation_id` — the extracted text is fed through `rag_service.update_db_files()` as a synthetic `<name>.handwritten-ocr.txt` file, indexed identically to a normal upload. Off by default.

**Not wired into automatic ingestion** — confirmed via grep: zero references to `handwritten_ocr_service` inside `loaders/`, `rag_service.py`, or `routes/upload.py`.

---

## 11. Audio / Whisper Pipeline

```
Browser mic
  │ MediaRecorder({audio:true}), 100ms timeslices
  ▼
VoiceRecorder.tsx → Blob(type:"audio/webm")
  │
  ▼
api.ts::askVoice() → POST /api/chat/voice (multipart: audio, language, conversation_id)
  │
  ▼
routes/chat.py::chat_voice
  │ asyncio.to_thread(transcribe_audio, audio_bytes, language=lang_hint)
  ▼
services/audio_service.py::transcribe_audio
  1. Reject if raw bytes < 5000 bytes ("microphone may not be working")
  2. _convert_to_wav() — ffmpeg subprocess → 16kHz mono PCM16 WAV
     (retries once with bare "ffmpeg" if FFMPEG_PATH fails; rejects if
      converted file < 2000 bytes)
  3. _get_audio_rms_db() — ffmpeg volumedetect filter
     → reject if mean_db < SILENCE_THRESHOLD_DB (-60.0)      [silence detection]
  4. _get_model() — lazy singleton, whisper.load_model(WHISPER_MODEL_NAME="small")
  5. Language: if caller passed ar/en explicitly, force it; else
     _detect_language() — Whisper encoder-only pass (log-mel spectrogram,
     no decoding), Arabic favored via `ar_prob >= en_prob * 0.75`
  6. model.transcribe(path, language=resolved, initial_prompt=_initial_prompt(lang),
       fp16=False, task="transcribe",
       temperature=(0.0,0.2,0.4,0.6,0.8,1.0),   ← full retry schedule, not a single value
       condition_on_previous_text=False, beam_size=5, best_of=5)
  7. Reject if resulting text is empty ("No speech detected")
  8. finally: delete both temp files
  │
  ▼
stt_text  ──►  routes/chat.py::_run_agent(stt_text, language, conversation_id)
                (the SAME helper the plain-text /chat route uses — no separate
                 voice code path past this point)
  │
  ▼
JSON {answer, sources, stt_text, report}  ──►  ChatBox.handleVoice
                                                  (replaces the "Voice message…"
                                                   placeholder with stt_text,
                                                   then renders the answer as usual)
```

**Initial prompts:** Arabic gets a dialect-tuned (Egyptian) prompt instructing verbatim transcription, no translation, contextual correction of commonly-misheard words, and preservation of English technical terms; English gets a plain "transcribe exactly, don't translate, keep technical English terms" prompt.

**Error handling:** every failure path (too-small audio, ffmpeg failure, silent audio, no speech detected) raises a plain `RuntimeError` with a human-readable message, caught in `routes/chat.py::chat_voice` and turned into `HTTPException(422, ...)`.

**TTS:** **Confirmed absent.** Grepped `tts|text-to-speech|speak|SpeechSynthesis` (case-insensitive) across the entire repository — the only hit is the unrelated substring "speak" inside "Arabic-speaking" in a doc file. No `/tts` route is mounted in `main.py`, no `SpeechSynthesisUtterance`/`window.speechSynthesis` anywhere in the frontend, no audio-playback element tied to AI answers. The pipeline is one-directional: mic → STT → text → agent → **text** answer.

---

## 12. Embedding System

- **Model:** `intfloat/multilingual-e5-large` (default, `EMBEDDING_MODEL`), an E5 model — uses `"query: "`/`"passage: "` instructional prefixes transparently inside `LocalEmbeddings` (rest of the codebase calls `embed_query`/`embed_documents` with plain text, unaware of the prefixing).
- **Provider:** local, on-device `sentence-transformers` — `EMBEDDING_PROVIDER` only supports `"local"` (anything else warns and falls back).
- **Device:** `EMBEDDING_DEVICE` (`auto`/`cpu`/`cuda`) via `utils/device.py::resolve_device()` — shared with the cross-encoder reranker, cached process-wide after first resolution.
- **Initialization/singleton:** `embeddings_provider.get_embeddings()` — `SentenceTransformer` loaded exactly once, module-level singleton, reused for every subsequent call from any request.
- **Query embeddings:** `embed_query()` (single) and `embed_queries()` (batched — used for the up-to-22 retrieval variants in one forward pass, deliberately, since concurrent single-item GPU calls measured ~10x slower than one batch of the same size, per `PROFILING.md`).
- **Document embeddings:** `embed_documents()` (batched, at ingestion time and inside MMR diversification/semantic/hybrid chunking).
- **Normalization:** `normalize_embeddings=True` (L2-normalized) — required for the cosine-distance MMR math (`1 - dot product`) and for Qdrant's `Distance.COSINE` collection config.
- **Multiple embedding passes per retrieval?** Yes — one batched pass for all query variants (`embed_queries`), plus (conditionally) one more batched `embed_documents` pass inside `_diversify()` for MMR over the reranked candidate pool, plus (if `CHUNKING_STRATEGY` is semantic/hybrid) additional passes at ingestion time only, not per query.
- **Stored:** in Qdrant (via `QdrantVectorStore.add_documents`). **Queried:** via `similarity_search_with_score_by_vector` against precomputed query vectors.

---

## 13. Vector Database (Qdrant)

- **Client:** `services/db_service.py::get_client()` — singleton `QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None, timeout=QDRANT_TIMEOUT_SECONDS)`.
- **Mode:** server (REST API) only — `QDRANT_URL` default `http://localhost:6333` (native dev) or `http://qdrant:6333` (Docker Compose service). **Embedded/file mode is explicitly not supported** (per the module docstring).
- **Collection:** `QDRANT_COLLECTION` (default `enterprise_docs`), single collection shared by every conversation.
- **Vector dimensions:** determined dynamically at collection-creation time by embedding a test string (`embeddings.embed_query("test")`) — not hardcoded; `multilingual-e5-large` produces 1024-dim vectors.
- **Distance metric:** `Distance.COSINE` (hardcoded in `ensure_collection`).
- **Schema safety:** if the collection already exists, `_check_existing_schema()` verifies vector size/distance match and **raises `RuntimeError`** on mismatch rather than silently recreating/deleting — an incompatible schema requires a human decision (e.g. after changing `EMBEDDING_MODEL`).
- **Metadata/payload:** every point's metadata includes `source`, `file_type`, `page`, `timestamp`, `conversation_id`, `document_id`, `chunk_index`, `total_chunks`, plus Excel-specific (`sheet_name`, `chunk_type`, `row_range`) or PDF-OCR-specific (`ocr_fallback`) fields as applicable. LangChain's `QdrantVectorStore` stores the whole metadata dict under the payload key `"metadata"` (so filter field paths are `"metadata.conversation_id"`, not top-level).
- **Upsert:** `vdb.add_documents(chunks)` (LangChain wrapper).
- **Search:** `similarity_search_with_score_by_vector(vector, k=RETRIEVER_K, filter=conv_filter)`.
- **Filtering:** every search/delete carries a `qdrant_models.Filter` on `metadata.conversation_id` (and, for document-scoped deletes/reindex, `metadata.document_id` too) — **there is no unfiltered "search everything" code path** (see §17, Document Isolation).
- **Batching:** ingestion batches the whole chunk list into one `add_documents` call per upload job (not chunked further at the Qdrant-call level).
- **Connection handling/retries:** every Qdrant network call goes through `with_retries()` (`QDRANT_CONNECT_RETRIES`=5 attempts, `QDRANT_RETRY_DELAY_SECONDS`=2.0s fixed delay), used for client construction, collection checks/creation, and deletes; `is_available()` (health endpoint) is a single-shot probe with no retries by design.

---

## 14. LLM Architecture

**Single provider: Groq** (`services/llm_provider.py` is the only module importing the `groq` SDK). No local/self-hosted LLM, no other provider anywhere in the codebase.

| Model setting | Default | Used for |
|---|---|---|
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Answer generation, translation, query rewriting, summarize, compare, respond-from-memory, memory fact extraction (via `rag_service.get_llm()`), report map/reduce (`report_service.get_llm()`) |
| `AGENT_MODEL` | `llama-3.1-8b-instant` | Planner action-selection step only (`agent/llm.py`) — smaller/faster model since it's a structured routing decision, not open-ended generation |

**Distinguishing call sites:**
- **Planner LLM calls:** `agent/llm.py::AgentLLM.invoke` → `GroqLLM.chat(messages, json_mode=True)` — the only `json_mode` call site in the app.
- **Query rewriting:** `rag_service._rewrite_query` (combined typo-fix + synonym-expansion, single call, `lru_cache(256)`).
- **Translation:** `rag_service._translate` (`lru_cache(512)`).
- **Final answer generation:** `rag_service.generate_answer`/`generate_answer_stream` → `build_prompt_with_memory` → `llm.invoke`/`llm.stream`.
- **Summarize/compare/respond:** `rag_service.summarize`/`compare`/`answer_from_memory` (+ `_stream` variants) — each its own dedicated bilingual prompt.
- **Memory fact extraction:** `memory/fact_extractor.py::extract_facts` via `memory/llm_adapter.py::LLMTextGenerator.generate` → (lazily) `rag_service.get_llm()`.
- **Report generation:** `report_service.py::_map_extract` (per-slice, concurrent pool of 5) and `_reduce_narrative` (4 concurrent calls) and `_topic_is_covered` (a yes/no relevance gate).

**Streaming vs non-streaming:** `GroqLLM.chat`/`.invoke` are non-streaming (single response); `.stream`/`.stream_chat` yield text deltas — used exclusively by `agent.run_stream()`'s terminal-action dispatch (`generate_answer_stream`, `summarize_stream`, `compare_stream`, `answer_from_memory_stream`), i.e. only for `/ws/chat`.

**Generation parameters:** `LLM_TEMPERATURE` (0.0), `LLM_MAX_TOKENS` (800), `LLM_TOP_P` (0.90) — applied uniformly to every `GroqLLM` instance regardless of which model.

**Retry behavior:** the planner (`agent/llm.py`) retries invalid JSON/schema up to 2x with a corrective follow-up message, then falls back deterministically. Every other Groq call site (`rag_service.py`, `report_service.py`, `memory/`) catches exceptions locally and degrades (empty string / error message in the answer / fail-open in `_topic_is_covered`) rather than retrying — **no automatic retry-on-transient-error exists for Groq calls outside the planner.**

**Debug instrumentation:** `llm_provider._log_outgoing_messages()` — gated by `AGENT_DEBUG` + DEBUG log level, logs the exact outgoing message list for every single Groq call in the app (one universal choke point).

---

## 15. Memory Architecture

**Two tiers, both per-conversation, coordinated by `memory/memory_manager.py::MemoryManager`:**

### Short-term (`memory/short_memory.py::ShortMemory`)
In-RAM only. `should_summarize()` triggers when `len(messages) > MEMORY_MAX_MESSAGES` (25) **or** `total_chars() > MEMORY_MAX_CHARS` (12000). Checked after every `add_message()` call (twice per turn, via `add_turn`).

### Long-term (`memory/summary_memory.py::FactStore` + `SummaryMemory`)
A capped, deduplicated list of discrete facts (`{text, category, importance, updated_at}`) — **not** a free-text paragraph.
- **Extraction:** LLM-based (`memory/fact_extractor.py::extract_facts`) — given existing facts + the recent message window, the model returns `{"facts":[...], "remove":[...]}`.
- **Merge/dedup:** `FactStore.merge()` — fuzzy match via `difflib.SequenceMatcher` at a 0.92 similarity threshold; a match **replaces** the existing fact (fresh `updated_at`) rather than duplicating; `remove_texts` entries drop matching facts.
- **Eviction:** `FactStore._cap()` — once over `MEMORY_MAX_FACTS` (40), evicts lowest `(importance, updated_at)` first.
- **Rendering:** `FactStore.render()` — recency + importance sorted, truncated to `MEMORY_SUMMARY_MAX_CHARS` (1200 chars).
- **Persistence:** `SummaryMemory` — one JSON file per conversation at `MEMORY_STORAGE_DIR/<sanitized_conversation_id>.json`, schema `{"version":2, "conversation_id", "facts":[...], "updated_at"}`. Verified directly against the real file `backend/memory_storage/f10ca108-b192-4c49-815e-3e8bef4017eb.json`. Legacy v1 `{"summary":...}` files are transparently wrapped into one synthetic fact on load.

### Trigger & background execution
`MemoryManager._summarize_async()`: when `should_summarize()` fires, short memory is **synchronously** trimmed to `MEMORY_KEEP_RECENT` (4) messages immediately (so the buffer can't keep growing while extraction runs), then a **daemon background thread** performs the actual LLM extraction + `FactStore.merge` + disk save — failures are caught and logged, never propagated to the user-facing turn.

### Where memory is read/injected
- **Planner prompt:** yes — `Agent._build_messages` injects `memory_manager.as_prompt_text()` into `USER_PROMPT`'s `{memory}` slot, visible to every planning decision including whether to retrieve at all.
- **RAG retrieval query itself:** **no** — `rag_service.retrieve()`/`RetrieveTool.run()` take no memory parameter; memory can only shape retrieval indirectly, through whatever query text the memory-aware planner chose.
- **Final answer-generation prompt:** yes — `generate_answer`/`generate_answer_stream` accept `memory: str` and inject it via `build_prompt_with_memory()`; `answer_from_memory`/`_stream` build a memory-only prompt (`_memory_only_prompt`, strictly scoped to greetings/small-talk/meta-conversation, explicitly forbidden from answering general-knowledge questions from the model's own training).

### Lifecycle
`agent/session.py` evicts idle `Agent` (and therefore `MemoryManager`) instances from the in-process registry after `AGENT_IDLE_TIMEOUT_SECONDS` (1800s) of inactivity with no in-flight request — this **only frees RAM**; the persisted `memory_storage/*.json` fact store is untouched, and a later request for the same `conversation_id` reloads long-term facts from disk into a fresh `Agent` (short-term memory starts empty).

---

## 16. WebSocket / Streaming Architecture

**Route:** `routes/ws.py::ws_chat` — `WS /ws/chat`, no `/api` prefix (mounted directly in `main.py`).

**Protocol** (JSON text frames both directions):
```
Client → Server:  {"query": "...", "language": "auto"|"ar"|"en", "conversation_id": "..."}
Server → Client:  {"type": "start"}
                  {"type": "status", "text": "..."}        (optional, e.g. report generation)
                  {"type": "token", "text": "..."}         (repeated, in order)
                  {"type": "done", "answer": "...", "sources": "...", "report"?: {...}}
                  {"type": "error", "message": "..."}
```

**Queue/thread interaction:** `agent.run_stream()` is a synchronous generator (the `groq` SDK is not async) — it is driven inside `asyncio.to_thread(_produce)`, where `_produce()` iterates the generator and pushes each yielded event onto an `asyncio.Queue` via `loop.call_soon_threadsafe`. The main coroutine (`_stream_answer`) awaits `queue.get()` in a loop and forwards each item to the WebSocket as JSON, until a sentinel `None` signals completion.

**Why the first token may be delayed:** everything before the first `"token"` frame — language detection, memory loading, the full planner loop (which itself may call `retrieve` one or more times, each involving query-variant generation with 2 concurrent Groq calls, batched embedding, concurrent Qdrant search, cross-encoder reranking, and MMR) — happens on the worker thread before the terminal action's token stream even starts. A `"status"` frame (e.g. "Generating the report...") is the only mid-flight signal for the one terminal action (`report`) that has no meaningful token stream at all.

**Context propagation:** the producer `asyncio.Task` is created (`asyncio.create_task`) **before** the `to_thread` hop specifically so `contextvars.Context` (carrying the `utils.timing` `RequestTimer`) is copied into the worker thread — otherwise per-request profiling would silently no-op for every WS request.

**Connection lifecycle:** one WebSocket connection can serve multiple sequential questions (the `while True: receive_text()` loop); `WebSocketDisconnect` is caught and logged, ending the handler. On the client side (`frontend/services/api.ts::streamChat`), a **fresh WebSocket is opened per call** (not reused across questions), despite the backend supporting reuse — confirmed via the frontend's own doc comment. The client also handles `onerror`/premature `onclose` as failure paths (`fail("Connection closed before a response was received.")`).

**Status events / error events:** `"status"` (informational, keeps the loading bubble but shows caption text) and `"error"` (terminates the turn, rendered as a red error bubble in `ChatBox.tsx`) are both distinct from the token stream itself.

---

## 17. Configuration Map

All settings are defined in `backend/config.py::Settings`, sourced from `.env` (see `.env.example`) via `python-dotenv`.

### LLM / Groq
| Var | Default | Controls |
|---|---|---|
| `GROQ_API_KEY` | `""` | Required for any Groq call — `llm_provider._get_client()` raises `RuntimeError` if unset |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Main generation/translation/rewrite/summarize/compare/memory/report model |
| `AGENT_MODEL` | `llama-3.1-8b-instant` | Planner action-selection model |
| `LLM_TEMPERATURE` | `0.0` | Generation randomness (all Groq calls) |
| `LLM_MAX_TOKENS` | `800` | Max output tokens per Groq call |
| `LLM_TOP_P` | `0.90` | Nucleus sampling |

### Embeddings
| Var | Default | Controls |
|---|---|---|
| `EMBEDDING_PROVIDER` | `local` | Only `"local"` supported |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | Sentence-transformers checkpoint |
| `EMBEDDING_DEVICE` | `auto` | CPU/CUDA for embedding model + cross-encoder reranker |

### Qdrant
| Var | Default | Controls |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Server endpoint |
| `QDRANT_API_KEY` | `""` | Optional auth |
| `QDRANT_COLLECTION` | `enterprise_docs` | Collection name |
| `QDRANT_TIMEOUT_SECONDS` | `10.0` | Client timeout |
| `QDRANT_CONNECT_RETRIES` | `5` | Retry attempts (`with_retries`) |
| `QDRANT_RETRY_DELAY_SECONDS` | `2.0` | Fixed delay between retries |

### RAG / Reranking
| Var | Default | Controls |
|---|---|---|
| `ENABLE_PDF_OCR_FALLBACK` | `true` | Whether `pdf_loader.py` ever invokes OCR at all |
| `MAX_UPLOAD_SIZE_MB` | `200` | Hard cap enforced incrementally during upload streaming |
| `RETRIEVER_K` | `8` | Candidates pulled per query variant |
| `RERANK_TOP_N` | `6` | Final reranked result count (chat) |
| `EXCEL_RERANK_TOP_N` | `12` | Widened result count when Excel chunks are present |
| `CONFIDENCE_THRESHOLD` | `0.05` | Coarse near-zero-overlap rejection threshold |
| `RERANK_USE_CROSS_ENCODER` | `true` | Enable/disable cross-encoder scoring |
| `CROSS_ENCODER_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Reranker checkpoint |
| `RERANK_ALPHA` | `0.6` | Cross-encoder vs lexical score weight |
| `RERANK_DIVERSIFY` | `true` | Enable MMR-lite reselection |
| `MMR_LAMBDA` | `0.7` | Relevance vs. diversity tradeoff |
| `MAX_CONTEXT_CHARS` | `6000` | Prompt context character budget |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `700` / `150` | Recursive splitter sizing (also derives Excel row-group sizing) |
| `QUERY_EXPANSION_ENABLED` | `true` | Synonym/concept query variant generation |
| `CHUNKING_STRATEGY` | `recursive` | `recursive`\|`semantic`\|`hybrid` |
| `SEMANTIC_CHUNK_*` / `HYBRID_*` | various | Tuning for the two alternate chunking strategies |
| `EXCEL_ROWS_PER_CHUNK_MIN/MAX`, `EXCEL_SUMMARY_SAMPLE_ROWS`, `EXCEL_MAX_ROWS_PER_SHEET` | `3`/`50`/`5`/`20000` | Excel ingestion sizing |

### OCR
| Var | Default | Controls |
|---|---|---|
| `TESSERACT_CMD` | `tesseract` | Tesseract binary path |
| `OCR_MAX_CONCURRENT_PAGES` | `4` | Bounded page-level parallelism |
| `OCR_MIN_TEXT_CHARS` | `20` | Scanned-page/document detection threshold; also the OCR-confidence length gate |
| `OCR_MIN_ALNUM_RATIO` | `0.6` | OCR-confidence alnum-ratio gate |
| `HANDWRITTEN_OCR_EN_MODEL` | `microsoft/trocr-small-handwritten` | English TrOCR checkpoint |
| `HANDWRITTEN_OCR_AR_MODEL` | `RayR1/trocr-base-arabic-handwritten` | Arabic TrOCR checkpoint |
| `HANDWRITTEN_OCR_MAX_NEW_TOKENS` | `256` | Generation length cap per line |
| `HANDWRITTEN_OCR_MAX_BATCH_SIZE` | `8` | Multi-line batch size |

### Whisper / FFmpeg
| Var | Default | Controls |
|---|---|---|
| `WHISPER_MODEL_NAME` | `small` | Whisper model size |
| `SILENCE_THRESHOLD_DB` | `-60.0` | Silence-rejection threshold |
| `FFMPEG_PATH` | `ffmpeg` | ffmpeg binary path |

### Upload
| Var | Default | Controls |
|---|---|---|
| `PROCESSED_FILES_REGISTRY` | `./processed_files.json` | Ingestion dedup registry path |
| `UPLOAD_FOLDER` | `./stored_files` | **Deprecated** — kept only so old `.env` files don't error; originals now go to MinIO |

### Agent
| Var | Default | Controls |
|---|---|---|
| `AGENT_MAX_ITERATIONS` | `6` | ReAct loop cap |
| `AGENT_DEBUG` | `false` | Per-step debug logging + Groq wire-message logging |
| `DEFAULT_CONVERSATION_ID` | `default` | Fallback id (not used in practice — every route requires an explicit id) |
| `AGENT_IDLE_TIMEOUT_SECONDS` | `1800` | Idle-Agent eviction threshold |
| `AGENT_CLEANUP_INTERVAL_SECONDS` | `300` | Eviction scan interval |

### Memory
| Var | Default | Controls |
|---|---|---|
| `MEMORY_MAX_MESSAGES` | `25` | Short-memory summarize trigger (count) |
| `MEMORY_KEEP_RECENT` | `4` | Messages kept after summarization |
| `MEMORY_WINDOW` | `6` | Recent messages rendered into prompts |
| `MEMORY_STORAGE_DIR` | `./memory_storage` | Fact-store JSON directory |
| `MEMORY_MAX_FACTS` | `40` | Fact-store cap |
| `MEMORY_SUMMARY_MAX_CHARS` | `1200` | Rendered fact-text char budget |
| `MEMORY_MAX_CHARS` | `12000` | Short-memory summarize trigger (chars) |

### MinIO / Reports
| Var | Default | Controls |
|---|---|---|
| `MINIO_ENDPOINT` / `MINIO_PUBLIC_ENDPOINT` | `localhost:9000` / falls back to endpoint | Internal vs. browser-facing host |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | `minioadmin` / `minioadmin` | Credentials |
| `MINIO_SECURE` | `false` | TLS toggle |
| `MINIO_BUCKET_UPLOADS` / `MINIO_BUCKET_REPORTS` | `doc-assistant-uploads` / `doc-assistant-reports` | Bucket names |
| `MINIO_PRESIGNED_EXPIRY` | `3600` | Presigned URL TTL (seconds) |
| `REPORT_MAP_CHUNK_CHARS` | `6000` | Report MAP-step slice size |
| `REPORT_FONT_DIR` | `<repo>/assets/fonts` | Arabic PDF font location |

### Debugging / Performance
| Var | Default | Controls |
|---|---|---|
| `LOG_REQUEST_PROFILE` | `true` | Per-request `utils.timing` stage breakdown logging |
| `LOG_LEVEL` | `INFO` | Global Python logging level |
| `LOG_RETRIEVAL_DEBUG` | `false` | Verbose retrieval debug (query variants, per-variant hits, scores) |

---

## 18. Dependency Map

```
routes/chat.py ──┬─► agent/session.py ─► agent/agent.py ─┬─► agent/llm.py ─► services/llm_provider.py (Groq)
                  │                                        ├─► agent/tools/* ─► services/rag_service.py
                  └─► services/audio_service.py (voice)     └─► memory/memory_manager.py ─┬─► memory/short_memory.py
                                                                                            └─► memory/summary_memory.py
                                                                                                  └─► memory/fact_extractor.py
                                                                                                        └─► memory/llm_adapter.py ─► services/rag_service.get_llm()

routes/ws.py ─► agent/session.py ─► agent/agent.py  (same fan-out as above, streaming variant)

services/rag_service.py ─┬─► loaders/registry.py ─┬─► loaders/pdf_loader.py ─► services/ocr_service.py
                          │                         ├─► loaders/image_loader.py ─► services/ocr_service.py
                          │                         ├─► loaders/docx_loader.py
                          │                         ├─► loaders/text_loader.py
                          │                         └─► loaders/excel_loader.py
                          ├─► services/db_service.py ─► Qdrant
                          ├─► services/embeddings_provider.py ─► utils/device.py
                          ├─► services/llm_provider.py (Groq)
                          └─► services/storage_service.py ─► MinIO

routes/upload.py ─┬─► utils/file_validation.py ─► loaders/registry.py
                   ├─► services/upload_jobs.py
                   └─► services/rag_service.py (update_db_files / reindex_document / delete_document)

routes/ocr.py ─► services/handwritten_ocr_service.py
              └─► services/rag_service.update_db_files  (only if index=true)

routes/reports.py ─► services/report_service.py ─┬─► services/rag_service.py (get_document_pages, retrieve, detect_language)
                                                    ├─► services/llm_provider.py (Groq)
                                                    └─► services/storage_service.py (MinIO)

agent/tools/report_tool.py ─► services/report_service.py + services/rag_service.py

frontend/services/api.ts ─► [every backend route above, via REST + WS]
  └─► frontend/components/{ChatBox,UploadBox,VoiceRecorder,HandwrittenOcrModal}.tsx
```

---

## 19. Data Flow

| Transition | Shape of the data |
|---|---|
| User question → planner input | Raw string + `ExecutionContext` (language, documents-so-far, observations, retrieved_questions) rendered into `USER_PROMPT` |
| Planner → tool call | `AgentAction` (Pydantic discriminated union: `{thought, action, arguments}`) |
| Tool call → query variants | `str` question → `List[str]` (≤22 variants: original, normalized, typo-corrected, synonym alternatives, translated, raw-question anchor) |
| Query variants → embeddings | `List[str]` → `List[List[float]]` (one batched call, L2-normalized, 1024-dim for the default model) |
| Embeddings → Qdrant results | vectors → `List[(Document, score)]` per variant, tagged with `_vector_score`/`_matched_variant` metadata |
| Qdrant results → reranked documents | deduped `List[Document]` → blended-score-sorted, MMR-reselected `List[Document]` + parallel debug-info list |
| Reranked documents → context | `List[Document]` → one `"[Chunk N | source | page/sheet-range]\n<text>"`-labeled, char-budget-trimmed string |
| Context → LLM prompt | context string + question + language + memory text → one bilingual prompt string (`build_prompt_with_memory`) |
| LLM prompt → streamed tokens | Groq SSE deltas → `{"type":"token","text":str}` frames (WS) or accumulated into one `str` (HTTP) |
| Final turn → memory write | `(question, final_answer)` → `ShortMemory` message pair; eventually → LLM-extracted `{facts:[...], remove:[...]}` → `FactStore` |
| Agent-facing chunk dict shape (`rag_service.retrieve()` output) | `{"id": "<source>::<page>::<chunk_index>::<md5[:8]>", "text": str, "metadata": {source, title, document_type, page, chunk_index, conversation_id, document_id, relevance_score, [excel fields]}}` |

---

## 20. Concurrency / Performance Architecture

| Mechanism | Where | Purpose |
|---|---|---|
| `asyncio.to_thread` | `routes/chat.py` (`_run_agent`, `transcribe_audio`), `routes/ws.py` (`_produce`), `routes/upload.py` (`_ingest_job`), `routes/ocr.py` (`_run_ocr`) | Offload synchronous/blocking Groq/Whisper/CPU work off the event loop |
| `asyncio.Queue` + `call_soon_threadsafe` | `routes/ws.py::_stream_answer` | Relay generator events from a worker thread back to the async WS send loop |
| `ThreadPoolExecutor` (via `utils.timing.run_concurrent_ctx`) | `rag_service._run_concurrent` (query rewrite + translate; concurrent per-variant Qdrant searches) | Parallelize independent Groq calls and I/O-bound Qdrant lookups, while preserving `contextvars` for the timing profiler |
| `ThreadPoolExecutor` (bounded, `OCR_MAX_CONCURRENT_PAGES`) | `ocr_service.perform_ocr_pdf_bytes`/`perform_ocr_pdf_pages_bytes` | Page-level OCR parallelism, capped regardless of page count |
| `ThreadPoolExecutor` (`MAP_EXTRACT_CONCURRENCY=5`, and a 4-worker pool for narrative reduce) | `report_service.build_report_data`/`build_topic_report_data` | Parallel per-slice MAP extraction + parallel REDUCE narrative generation |
| Batched embedding calls | `embeddings_provider.embed_queries`/`embed_documents` | One forward pass instead of N concurrent single-item calls (measured ~10x faster on GPU) |
| `threading.Thread(daemon=True)` | `memory/memory_manager.py::_summarize_async`, `agent/session.py::_cleanup_loop` | Background fact extraction (per-turn) and idle-Agent eviction (long-running daemon) |
| `threading.Lock` | `agent/agent.py` (`_activity_lock`/`_in_flight`), `agent/session.py` (`_lock`), `handwritten_ocr_service.py` (`_lock`, `_service_lock` — double-checked-locking model loads), `ocr_service.perform_ocr_pdf_pages_bytes` (result-dict lock) | Guard shared mutable state across threads |
| Singletons (lazy, module-level) | `embeddings_provider._embeddings`, `db_service._client`, `llm_provider._shared_llm`/`_shared_agent_llm`, `rag_service._cross_encoder`, `handwritten_ocr_service._service`/per-language models, `audio_service._model` | Avoid reloading heavy ML models / reconnecting clients per request |
| `lru_cache` | `rag_service._translate` (512), `_rewrite_query` (256) | Avoid repeat Groq calls for identical (query, lang) pairs within process lifetime |
| Synchronous (no concurrency) | Most single-document loaders (`docx_loader`, `text_loader`), `db_service.with_retries` (sequential retry, not parallel) | — |

**Dead/unused code candidates found and verified:**
- `services/ocr_service.py::extract_text()` — defined, dispatches by extension, but **grep found zero call sites** anywhere outside the file itself. The runtime pipeline calls `perform_ocr_pdf_bytes`/`perform_ocr_pdf_pages_bytes`/`perform_ocr_image_bytes` directly from `loaders/`. **Confidence: high** (dedicated OCR-subsystem investigation grepped the whole `backend/` tree). Status: Dead code / legacy convenience wrapper.
- `services/audio_service.py::transcribe_audio_path()` — defined, calls `transcribe_audio()` internally, but **no external caller found** anywhere in the repo (the one route that transcribes audio, `chat_voice`, passes bytes directly to `transcribe_audio`). **Confidence: high**. Status: Dead code / unused helper (plausibly kept for scripting/manual testing convenience, but not verified as intentional).
- `frontend/services/api.ts::askQuestion()` — exported, wraps `POST /api/chat`, but `ChatBox.tsx` uses `streamChat()` (the WebSocket) for all text-submit flows; no caller of `askQuestion` was found among the frontend files investigated. **Confidence: medium** (could not exhaustively grep every frontend file, but all AI-pipeline-relevant components were read). Status: Unclear / needs verification — possibly a leftover from a pre-WebSocket design, or intentionally kept for callers outside the read file set.
- `frontend/services/api.ts::generateReport()` — exported, wraps `POST /api/reports/generate`, but no UI trigger was found calling it; reports currently appear to reach the UI only via the chat `done`/response payload's `report` field (populated by the agent's `report` tool). **Confidence: medium**. Status: Unclear / needs verification — the backend route (`routes/reports.py`) is real and independently reachable (e.g. via direct API call or a future UI), so this is not dead on the backend side.
- `backend/tmp_unused/` — an empty directory with no files. Not further investigated (nothing to investigate).

**Not dead (confirmed real, just non-automatic):** `services/handwritten_ocr_service.py` and `routes/ocr.py` — a fully wired, real feature (frontend modal → dedicated route → service), just intentionally not part of the automatic upload pipeline.

---

## 21. Error Handling

| Subsystem | What can fail | Where caught | Fallback | User-visible result |
|---|---|---|---|---|
| Groq (planner) | Invalid JSON, schema mismatch, network/API error | `agent/llm.py::AgentLLM.invoke` | Retry ×2 with corrective message, then deterministic `RetrieveAction` fallback | Turn proceeds normally (possibly with a forced retrieve) |
| Groq (generation/translate/rewrite/summarize/compare/memory/report) | Any exception | Locally in each `rag_service.py`/`report_service.py`/`memory/*` function (`try/except`) | Empty string / `"Error generating answer: {e}"` in the answer text / fail-open (e.g. `_topic_is_covered` assumes relevant on error) | Chat still returns a response (possibly an error string as the "answer"), rarely a 500 |
| Qdrant | Connection refused, container still starting, transient network error | `db_service.with_retries` (5 retries, 2s delay) | Bounded retry, then re-raise | Startup: app still starts, `is_ready()` reports false until a later call succeeds. Mid-request: `HTTPException(500)` via route-level try/except |
| Embeddings/cross-encoder | Model fails to load (offline, not cached) | `rag_service._get_cross_encoder` | Permanent fallback to lexical-only reranking for the process lifetime (no retry) | Retrieval still works, just without semantic reranking |
| OCR (Tesseract) | Any per-page/per-strategy exception | Inside `ocr_service.py`'s own functions | Returns `""`/skips that page or strategy; last-resort single Tesseract call for images | Missing/empty text for that page rather than an ingestion failure |
| OCR (TrOCR) | Bad image, unsupported language, model load/inference failure | `routes/ocr.py` (`InvalidImageError`→400, `UnsupportedLanguageError`→400, `OCRModelError`→500) | None — surfaced directly | HTTP error to the modal, shown as `error` state |
| Whisper/ffmpeg | ffmpeg missing/fails, silent audio, empty transcription, audio too small | `services/audio_service.py` (raises `RuntimeError` with specific messages) | ffmpeg: one retry with bare `"ffmpeg"` PATH lookup | `routes/chat.py::chat_voice` converts to `HTTPException(422, ...)` |
| Upload validation | Wrong extension, corrupt/mismatched content, empty file, oversized file | `utils/file_validation.py` + `routes/upload.py` (`_stream_upload_to_temp_file`, `_validate_temp_file`) | None — rejected upfront, before any job is created | `HTTPException(400/413)` |
| Ingestion (background job) | Any exception during parse/chunk/embed | `routes/upload.py::_ingest_job`'s `try/except`, categorized by `_categorize_error(stage, exc)` | None — job marked `"error"` | Frontend polls, shows the categorized error + "Try again" retry button |
| MinIO | Package missing, server unreachable | `services/storage_service.py` (every function) | Raises `StorageUnavailableError`; ingestion proceeds without the original file being downloadable | `/health` reports `"unreachable"`; report/download routes return `HTTPException(503)` |
| Agent (overall) | Max iterations exhausted | `agent/agent.py::_run_impl`/`_run_stream_impl` (`for...else`) | Forces a final `generate` call from whatever context exists | User still gets an answer, just possibly less complete |
| WebSocket | Client disconnect mid-stream, malformed JSON frame | `routes/ws.py` (`WebSocketDisconnect` caught; JSON errors → `{"type":"error"}` per-message, connection stays open) | None needed — per-message error handling | `{"type":"error","message":...}` frame; connection only closes on actual disconnect |

**Retried:** only Qdrant operations (via `with_retries`) and ffmpeg invocation (one bare-PATH retry) and the planner's structured-output validation (2 retries). **Not retried:** every other Groq call, OCR strategy escalation (that's a deliberate tiered *sweep*, not a retry-on-failure), MinIO operations, embedding calls.

---

## 22. Dead / Unused Code

See the consolidated list with confidence levels in §20 (Performance/Concurrency section, "Dead/unused code candidates"). Summary:

| Candidate | File | Confidence | Evidence |
|---|---|---|---|
| `extract_text()` | `backend/services/ocr_service.py` | High | Grepped whole `backend/` tree; only self-references |
| `transcribe_audio_path()` | `backend/services/audio_service.py` | High | Grepped whole `backend/` tree; only internal delegation to `transcribe_audio()` |
| `askQuestion()` | `frontend/services/api.ts` | Medium | No caller among all AI-pipeline frontend files read; `ChatBox.tsx` uses `streamChat()` exclusively for text |
| `generateReport()` | `frontend/services/api.ts` | Medium | No UI trigger found; backend route is real and independently reachable, so not dead end-to-end |
| `backend/tmp_unused/` | directory | N/A | Empty — no files to assess |

Nothing else in the AI layer was found to be unreferenced; every other file/function traced back to a live call chain from a route or from another actively-used module.

---

## Architecture Observations

*(Documentation only — nothing below was fixed or changed.)*

**MEDIUM**
- `services/ocr_service.py::extract_text()` (lines 307-317) is a dead convenience dispatcher — the real pipeline (`loaders/pdf_loader.py`, `loaders/image_loader.py`) calls the lower-level `perform_ocr_*` functions directly, bypassing it entirely. Low risk since it's unreferenced, but it's a second, silently-diverging "front door" to the OCR service that a future contributor could mistakenly wire up instead of the actual pipeline.
- `services/audio_service.py::transcribe_audio_path()` (lines 225-227) has no call site anywhere in the repo — the only STT entry point in use is `transcribe_audio(bytes, ...)` from `routes/chat.py::chat_voice`.
- `frontend/services/api.ts::askQuestion()` (lines ~112-124) and `generateReport()` (lines 268-275) are exported with no traced caller among the AI-pipeline frontend files. `askQuestion` in particular duplicates functionality `streamChat()` already covers for the same UI (`ChatBox.tsx`) — two independent client-side code paths for "ask a text question" (HTTP vs WS) increase the surface a future change has to keep in sync, even though only one is currently exercised.
- `agent/tools/report_tool.py::run()` (whole-document report path, lines 53-140) resolves its target document via `rag_service.list_stored_files()` with `conversation_id=None` — i.e. across **every** conversation's uploads, not just the current one — while the topic-scoped path and all retrieval elsewhere in the app are strictly conversation-scoped (see Document Isolation, §11/§13). This is explicitly documented in the code as a deliberate, known limitation rather than an oversight, but it is a real inconsistency in the isolation model: a user could, in principle, generate a report against a document uploaded by a different conversation if they guess/know its exact filename.

**LOW**
- `services/handwritten_ocr_service.py::HandwrittenOCRService._resolve_device()` (lines 309-315) duplicates the CUDA/CPU auto-detection logic already centralized in `utils/device.py::resolve_device()`, but reimplements it independently and offers no manual override equivalent to `EMBEDDING_DEVICE`. Two independent device-resolution code paths for what is conceptually the same decision (place a local torch model on GPU if available) is a minor duplication; a user wanting to force TrOCR onto CPU (e.g. to reserve GPU memory for the embedding model) has no config knob to do so.
- `routes/chat.py::ChatRequest`/`routes/ws.py` both independently guard against a missing/empty `conversation_id` with near-identical inline logic and comments referencing the same historical bug ("Issue 2" — unrelated conversations merging). The guard itself is correct and important, but it's duplicated rather than shared (e.g. via a common dependency/validator), so a third future entry point could omit it.
- `backend/tmp_unused/` is an empty, presumably vestigial directory left in the repository tree; harmless but is repo clutter that could confuse a new contributor into thinking something is missing from it.

**INFO**
- The confidence-threshold retrieval filter (`CONFIDENCE_THRESHOLD`, `rag_service._retrieve`) is explicitly documented in its own comments as a coarse heuristic that cannot reliably distinguish "on-topic but loosely worded" from "genuinely unrelated" — the real grounding guarantee is the LLM prompt rule in `build_prompt()`. This is a conscious, well-documented tradeoff, not a defect, but worth knowing when reasoning about why an off-topic-looking question sometimes still reaches the LLM (and correctly gets refused there instead).
- The planner model (`AGENT_MODEL`, default `llama-3.1-8b-instant`) is deliberately smaller/faster than the generation model (`GROQ_MODEL`) for latency reasons — this is a real, intentional coupling between routing quality and cost/latency that shows up as `agent.py::_correct_premature_terminal`'s deterministic backstop existing specifically to compensate for the smaller model's occasional misclassification of "explain X"-style imperative phrasings as small talk.
- Report generation (`report_service.py`) and memory fact extraction (`memory/fact_extractor.py`) both perform multiple LLM calls per operation (report: 1 per MAP slice + 4 REDUCE calls + 1 relevance-gate call for topic reports; memory: 1 extraction call per summarization trigger) — all already parallelized where independent (bounded thread pools), so this is a documented cost-of-doing-business rather than an obvious inefficiency.
- Whole-document PDF OCR (`ocr_service.perform_ocr_pdf_bytes`) collapses an entire scanned document into a **single** `Document` (one chunk pre-split), unlike the mixed-document path which preserves per-page `Document` boundaries. This means whole-document metadata like per-page citation (`page` field in retrieved-chunk sources) is coarser for fully-scanned PDFs than for text-based or mixed ones — a real, observable behavior difference a user might notice as "page numbers in sources are less precise for this particular scanned file," not a bug, just a byproduct of the whole-blob OCR path's design.

---

## 24. Final Architecture Diagram

```
                                    ┌────────────────────────┐
                                    │        Browser          │
                                    │  ChatBox / UploadBox /  │
                                    │  VoiceRecorder / OCR     │
                                    │  Modal / ReportCard      │
                                    └────────────┬─────────────┘
                                                 │
                            ┌────────────────────┼─────────────────────┐
                            │ HTTP (REST)         │ HTTP (multipart)     │ WebSocket
                            ▼                     ▼                     ▼
                   /api/chat, /voice,     /api/upload, /ocr/      /ws/chat
                   /reset, /reports,      handwritten,
                   /stored-files,         /documents/{id}
                   /health
                            │                     │                     │
                            ▼                     ▼                     ▼
                   routes/chat.py         routes/upload.py,      routes/ws.py
                                           routes/ocr.py
                            │                     │                     │
                            │            ┌────────┴─────────┐           │
                            │            ▼                  ▼           │
                            │   services/rag_service    services/       │
                            │   .update_db_files()       handwritten_   │
                            │            │                ocr_service   │
                            │            ▼                              │
                            │   loaders/registry.py ──► pdf_loader /    │
                            │            │              docx_loader /   │
                            │            │              text_loader /   │
                            │            │              image_loader /  │
                            │            │              excel_loader    │
                            │            │                    │         │
                            │            │                    ▼         │
                            │            │           services/ocr_      │
                            │            │           service.py         │
                            │            │           (Tesseract)        │
                            │            ▼                              │
                            │   embeddings_provider ──► Qdrant           │
                            │                                            │
                            └──────────────┬─────────────────────────────┘
                                            ▼
                                  agent/session.py::get_agent(conversation_id)
                                            │
                                            ▼
                                  ┌──────────────────────┐
                                  │   agent/agent.py       │
                                  │   (ReAct loop)          │
                                  └──────────┬──────────────┘
                                            │
                    ┌───────────────────────┼────────────────────────┐
                    ▼                       ▼                        ▼
           agent/llm.py            agent/tools/*.py         memory/memory_manager.py
           (planner, Groq          (retrieve/generate/       ├─ short_memory.py (RAM)
            AGENT_MODEL)            summarize/compare/        └─ summary_memory.py
                                     respond/report)              (fact store, JSON disk)
                                            │
                                            ▼
                                  services/rag_service.py
                    ┌───────────────────────┼────────────────────────┐
                    ▼                       ▼                        ▼
          embeddings_provider.py    services/db_service.py    services/llm_provider.py
          (local sentence-           (Qdrant client,           (Groq, GROQ_MODEL)
           transformers)              conversation-filtered)
                    │                       │                        │
                    └───────────► Qdrant ◄──┘                        │
                                     │                                 │
                                     ▼                                 │
                        cross-encoder reranker (sentence_transformers) │
                                     │                                 │
                                     ▼                                 │
                          MMR-lite diversification                    │
                                     │                                 │
                                     ▼                                 │
                          build_prompt_with_memory() ◄─────────────────┘
                                     │
                                     ▼
                          llm.invoke() / llm.stream()  (Groq)
                                     │
                    ┌────────────────┴─────────────────┐
                    ▼                                    ▼
         HTTP: full JSON response              WS: token-by-token frames
         {answer, sources, stt_text, report}    {"type":"token"/"done"/...}
                    │                                    │
                    └────────────────┬───────────────────┘
                                     ▼
                          Browser renders AnswerBox +
                          SourceBox + ReportCard

  ── Separate, parallel branches ──
  Voice:   Browser mic → VoiceRecorder → /api/chat/voice → audio_service.py
           (ffmpeg + Whisper) → stt_text → routes/chat.py::_run_agent (same as above)
           [NO TTS — text-only answer back]

  Handwritten OCR: HandwrittenOcrModal → /api/ocr/handwritten →
           handwritten_ocr_service.py (TrOCR) → text (+ optional opt-in indexing
           into the same RAG pipeline via update_db_files)
```

---

## 25. Complete File Index

| File | Purpose | Main Responsibility | Used By | Status |
|---|---|---|---|---|
| `backend/config.py` | Central settings | Every env var, typed | Every backend module | Core |
| `backend/main.py` | FastAPI app | Router mounting, CORS, startup DB attach | uvicorn | Core |
| `backend/agent/agent.py` | ReAct loop | Planning, tool dispatch, streaming, memory glue | routes/chat.py, routes/ws.py | Core |
| `backend/agent/llm.py` | Planner LLM wrapper | Structured JSON action selection + fallback | agent/agent.py | Core |
| `backend/agent/prompt.py` | Planner prompts | System/user prompt templates | agent/agent.py | Core |
| `backend/agent/registry.py` | Tool factory | Builds per-conversation tool dict | agent/agent.py | Core |
| `backend/agent/schemas.py` | Pydantic schemas | Action/context data models | agent/*, agent/tools/* | Core |
| `backend/agent/session.py` | Agent registry | Per-conversation Agent lifecycle + idle eviction | routes/chat.py, routes/ws.py | Core |
| `backend/agent/tools/retrieve_tool.py` | Retrieval tool | Wraps rag_service.retrieve | agent/registry.py | Core |
| `backend/agent/tools/generate_tool.py` | Answer tool | Final answer from docs+memory | agent/registry.py | Core |
| `backend/agent/tools/summarize_tool.py` | Summarize tool | Summarize retrieved docs | agent/registry.py | Core |
| `backend/agent/tools/compare_tool.py` | Compare tool | Compare retrieved docs | agent/registry.py | Core |
| `backend/agent/tools/respond_tool.py` | Memory-only tool | Small-talk/meta answers | agent/registry.py | Core |
| `backend/agent/tools/report_tool.py` | Report tool | PDF report generation dispatch | agent/registry.py | Core |
| `backend/services/rag_service.py` | RAG engine | Ingestion, retrieval, reranking, generation | agent/*, routes/*, report_service.py | Core |
| `backend/services/llm_provider.py` | Groq wrapper | Sole Groq SDK entry point | rag_service, agent/llm.py, report_service, memory | Core |
| `backend/services/embeddings_provider.py` | Local embeddings | Sentence-transformers singleton | rag_service, db_service | Core |
| `backend/services/db_service.py` | Qdrant wrapper | Client, collection mgmt, retries | rag_service | Core |
| `backend/services/report_service.py` | Report generation | Map-reduce + PDF rendering | report_tool.py, routes/reports.py | Core |
| `backend/services/ocr_service.py` | Printed OCR | Tesseract tiered strategy/PSM sweep | loaders/pdf_loader.py, loaders/image_loader.py | Core (`extract_text` dead) |
| `backend/services/handwritten_ocr_service.py` | Handwritten OCR | TrOCR + line segmentation | routes/ocr.py | Supporting (real, opt-in) |
| `backend/services/audio_service.py` | STT | ffmpeg + Whisper transcription | routes/chat.py | Core (`transcribe_audio_path` dead) |
| `backend/services/storage_service.py` | MinIO wrapper | Object storage for uploads/reports | rag_service, report_service, routes/upload.py | Core |
| `backend/services/upload_jobs.py` | Job tracker | In-memory upload job status | routes/upload.py | Core |
| `backend/loaders/registry.py` | Loader dispatch | Extension → loader mapping | rag_service, file_validation | Core |
| `backend/loaders/base.py` | Loader helpers | make_meta, clean_text | all loaders | Core |
| `backend/loaders/pdf_loader.py` | PDF parsing | Text extraction + OCR fallback decision | loaders/registry.py | Core |
| `backend/loaders/docx_loader.py` | DOCX parsing | Docx2txtLoader wrapper | loaders/registry.py | Core |
| `backend/loaders/text_loader.py` | Text/JSON parsing | TXT/MD/JSON loading | loaders/registry.py | Core |
| `backend/loaders/image_loader.py` | Image parsing | Always-OCR wrapper | loaders/registry.py | Core |
| `backend/loaders/excel_loader.py` | Excel/CSV parsing | Pre-sized sheet/row chunking | loaders/registry.py | Core |
| `backend/memory/short_memory.py` | Short-term memory | In-RAM message window | memory_manager.py | Core |
| `backend/memory/llm_adapter.py` | LLM decoupling | .generate() interface for fact_extractor | fact_extractor.py, memory_manager.py | Core |
| `backend/memory/fact_extractor.py` | Fact extraction | LLM-based structured fact parsing | memory_manager.py | Core |
| `backend/memory/summary_memory.py` | Long-term memory | FactStore + JSON persistence | memory_manager.py | Core |
| `backend/memory/memory_manager.py` | Memory coordinator | Agent's single memory interface | agent/agent.py, generate/respond tools | Core |
| `backend/routes/chat.py` | HTTP chat | /chat, /chat/voice, /chat/reset | main.py | Core |
| `backend/routes/ws.py` | WS chat | /ws/chat streaming | main.py | Core |
| `backend/routes/upload.py` | Upload API | Upload/status/reindex/delete/list | main.py | Core |
| `backend/routes/ocr.py` | Handwritten OCR API | /ocr/handwritten | main.py | Core |
| `backend/routes/reports.py` | Report API | Generate/download PDF reports | main.py | Core |
| `backend/routes/health.py` | Health check | Qdrant/MinIO reachability | main.py | Core |
| `backend/utils/device.py` | Device resolution | CPU/CUDA for embeddings + cross-encoder | embeddings_provider, rag_service | Core |
| `backend/utils/timing.py` | Request profiler | Per-stage latency breakdown | routes, rag_service, agent | Core |
| `backend/utils/file_validation.py` | Upload validation | Magic-byte/content checks | routes/upload.py | Core |
| `backend/scripts/evaluate_*.py` (×3) | OCR eval harnesses | Offline CER/WER benchmarking | manual execution only | Experimental (not runtime) |
| `backend/tests/*.py` (×5) | Test suite | Pytest coverage | pytest runner only | Supporting (QA) |
| `frontend/app/page.tsx` | App shell | Sidebar, health check, modal trigger | Next.js router | Core |
| `frontend/app/layout.tsx` | Root layout | HTML shell, metadata | Next.js router | Core |
| `frontend/components/ChatBox.tsx` | Chat UI | Streaming orchestration, message state | app/page.tsx | Core |
| `frontend/components/AnswerBox.tsx` | Answer rendering | Streaming text + typing indicator | ChatBox.tsx | Core |
| `frontend/components/SourceBox.tsx` | Citation rendering | Parses "Sources: A \| B" string | ChatBox.tsx | Core |
| `frontend/components/UploadBox.tsx` | Upload UI | Drag/drop, progress, file list | app/page.tsx | Core |
| `frontend/components/VoiceRecorder.tsx` | Voice capture | MediaRecorder → Blob | ChatBox.tsx | Core |
| `frontend/components/HandwrittenOcrModal.tsx` | OCR modal | Standalone handwritten-OCR UI | app/page.tsx | Core |
| `frontend/components/ReportCard.tsx` | Report download | Renders download link | ChatBox.tsx | Core |
| `frontend/components/ui/*.tsx` (×4) | UI primitives | Badge/Card/EmptyState/Skeleton | various | Supporting |
| `frontend/lib/conversation.ts` | Conversation identity | sessionStorage UUID | ChatBox.tsx | Core |
| `frontend/lib/fileTypeMeta.ts` | File-type badges | Icon/color/label mapping | UploadBox.tsx | Supporting |
| `frontend/services/api.ts` | Backend I/O | REST + WebSocket client, all endpoints | every AI-facing component | Core (2 exports unclear/unused) |
