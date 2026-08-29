# AI Document Assistant

An **agentic, bilingual (Arabic/English) Retrieval-Augmented Generation system**
for chatting with your own documents — LLM generation via the **Groq API**,
**Qdrant** vector storage (run as its own Docker container, server mode —
see [`docker-compose.yml`](docker-compose.yml)), and a configurable
embeddings provider.

Upload PDFs, Word docs, Excel spreadsheets/CSVs, text, JSON, or scanned
images; ask questions by typing or by voice; get grounded, cited answers
from an autonomous agent that decides for itself when to search your
documents, when to rely on conversation memory, and when to summarize or
compare what it has found.

---

## What's New: Handwritten OCR (Arabic + English, free & local)

- ✍️ **Handwritten OCR** — `POST /api/ocr/handwritten` recognizes Arabic or
  English handwriting from an uploaded image using Hugging Face `transformers`
  (TrOCR) — entirely local/free, no paid or external OCR API. Models
  (`microsoft/trocr-base-handwritten` for English,
  `RayR1/trocr-base-arabic-handwritten` for Arabic) download automatically on
  first use and are cached under the same Hugging Face cache the embedding/
  Whisper models already persist through (`backend_model_cache` in
  `docker-compose.yml` — no new volume needed). Separate from, and does not
  replace, the existing Tesseract-based printed-text OCR used by the upload
  pipeline. A "Handwritten OCR" button in the app header opens a modal to
  upload an image, pick a language, run OCR, and copy the result (right-to-left
  for Arabic). Extracted text can optionally be indexed into the same chunk →
  embed → Qdrant pipeline as any other upload. Full details, limitations
  (single-line images work best — no page-layout detector), and Docker/caching
  behavior: [`backend/HANDWRITTEN_OCR.md`](backend/HANDWRITTEN_OCR.md).

---

## What's New: Excel/CSV Ingestion, Modular Loaders, Redesigned UI

- 📊 **Excel/CSV ingestion** — `.xlsx`, `.xls`, and `.csv` files are now
  first-class citizens alongside PDFs. Every sheet is read (via pandas),
  turned into a sheet-summary chunk plus row-group chunks, tagged with
  `sheet_name` metadata, and stored/queried exactly like any other document.
  See `backend/loaders/excel_loader.py`.
- 🧩 **Modular loaders package** — per-file-type parsing (PDF, DOCX, TXT/MD,
  JSON, images, Excel) now lives in `backend/loaders/`, one module per type,
  dispatched through a single `loaders/registry.py` — no more inline
  if/elif branching in `rag_service.py`.
- 🎨 **Redesigned frontend** — refreshed cards, file-type badges, skeleton
  loaders, empty states, and upload progress, built on the existing design
  tokens. All backend-technology names (Groq, Qdrant, Whisper, etc.) have
  been removed from user-facing UI text.

### Previous round: Smarter Planner, Structured Memory, Better Retrieval

A five-part upgrade to the agent's reasoning and retrieval quality:

- 🧭 **Semantic intent recognition & coreference resolution** — the planner
  now recognises comparison/evaluation intent from natural phrasing ("which
  one is better?", "what changed?", "pros and cons?") instead of needing the
  literal word "compare", and resolves implicit references ("the two
  methods", "this document", "it") using the conversation's active document
  and memory instead of asking the user to repeat themselves. See
  `backend/agent/prompt.py` (Semantic Intent Recognition / Reference &
  Coreference Resolution sections) and `backend/agent/agent.py`.
- 🌐 **Deeper cross-language query expansion** — same-language synonym
  rephrasing and concept-level expansion (e.g. "advantages" → "benefits")
  now run on the user's original-language query, not just on translated
  variants, so evaluative/semantic questions match document wording that
  doesn't share the user's exact vocabulary. See `_concept_expand()` /
  `_query_variants()` in `backend/services/rag_service.py`.
- 🧠 **Structured long-term memory** — long-term memory is now a capped,
  deduplicated store of discrete facts (with category + importance) instead
  of one free-text paragraph rewritten from scratch each time. Near-duplicate
  facts are merged in code (not left to LLM instruction-following), the
  fewest-important/oldest facts are evicted once the cap is hit, and
  rendering is truncated to a character budget — bounding token usage
  regardless of conversation length. Old pre-upgrade summary files still
  load correctly (migrated into a single legacy fact). See
  `backend/memory/summary_memory.py`, `backend/memory/fact_extractor.py`.
- 🔎 **Cross-encoder reranking + diversity selection + context budgeting** —
  retrieval now blends a semantic cross-encoder relevance score with the
  existing lexical/bigram score (falling back to lexical-only if the model
  can't load), reselects the final chunk set for diversity (MMR-lite) so
  near-duplicate chunks don't crowd out distinct information, and trims
  context to a character budget so prompt size stays bounded. See `_rerank`,
  `_diversify`, `_trim_to_budget` in `backend/services/rag_service.py`.
- ✂️ **Concise-by-default answers** — answers are no longer forced into a
  fixed "Explanation:/Example:" shape with an example every time; the model
  now matches answer length to question complexity and only adds an example
  when it genuinely aids understanding or was explicitly requested. See
  `build_prompt()` in `backend/services/rag_service.py`.

---

## What's New: Streaming, MinIO Storage & PDF Reports

Three additions on top of the original agentic RAG pipeline:

- ⚡ **Real-time streaming answers (WebSockets)** — `/ws/chat` streams the
  agent's final answer token-by-token as it's generated, instead of
  waiting for the whole response. The frontend chat box renders it live,
  ChatGPT-style. See `backend/routes/ws.py`, `Agent.run_stream()` in
  `backend/agent/agent.py`, and `streamChat()` in `frontend/services/api.ts`.
- 🗄️ **MinIO object storage for uploads** — uploaded originals (PDF/DOCX/
  images/etc.) are stored in a MinIO bucket instead of local disk, so they
  survive container restarts and can be served via presigned URLs. See
  `backend/services/storage_service.py` and `docker-compose.yml` for a
  local MinIO instance.
- 📄 **Per-document PDF report generation — fully inside the chat, no
  Swagger, no buttons.** The agent's existing LLM-based intent planner
  (the same "thought → action" ReAct loop that already decides
  retrieve/generate/summarize/compare) now also recognises a `report`
  intent — semantically, not by keyword matching — from natural
  requests in English, Arabic, or mixed ("generate a report", "اعمل
  تقرير", "طلعلي PDF عن المستند"). It automatically figures out *which*
  uploaded document you mean (the one you're actively discussing, or
  the only one you've uploaded, or it asks you to pick if there are
  several), re-reads that document in full, and produces a
  comprehensive, professional multi-section PDF — cover page,
  auto-numbered table of contents, executive summary, introduction,
  detailed per-topic sections, key concepts, definitions, technical
  terms, important numbers, equations, referenced figures/tables, best
  practices, relationships between concepts, conclusion, and
  references, with page citations wherever possible. Large documents
  are processed section-by-section (map) and combined (reduce) instead
  of hitting token limits. The finished PDF is stored in MinIO and shows
  up right in the chat as a download card. See
  `backend/agent/tools/report_tool.py`, `backend/services/report_service.py`,
  and `backend/agent/prompt.py` (tool #6, "report").

**Setup — the whole stack (Qdrant + MinIO + backend + frontend) runs in
Docker; see [Section 5a](#5a-running-with-docker-compose-recommended) for
the full quickstart:**

```bash
cp .env.example .env                    # Compose-level vars (MinIO creds, Qdrant image tag)
cp backend/.env.example backend/.env     # fill in GROQ_API_KEY at minimum
docker compose up --build
# Frontend:      http://localhost:3000
# Backend:       http://localhost:8000/api/health
# MinIO console: http://localhost:9001  (minioadmin / minioadmin by default)
# Qdrant REST:   http://localhost:6333
```

> **MinIO is optional at runtime.** The backend still starts, and
> upload/chat/retrieval/memory all still work, even if the `minio` package
> isn't installed or the MinIO server isn't running — `GET /api/health`
> reports `"minio": "unreachable"` in that case instead of crashing.
> Without it, uploaded originals aren't stored (`download_url` comes back
> `null` for those files) and PDF report generation/download return
> `503 Service Unavailable`, since both need the original file bytes.
>
> **Qdrant is NOT optional** — it's the vector store retrieval/chat
> depend on. Unlike MinIO, the backend retries connecting to it on
> startup and after transient failures (see `services/db_service.py`),
> but degrades to "no documents retrievable" rather than crashing if it
> stays unreachable; `GET /api/health` reports `"qdrant": "unreachable"`
> in that case.

No extra setup is needed for streaming — `/ws/chat` is served by the same
FastAPI app as the REST routes.

---

## 1. Project Overview  

This project started as a straightforward RAG chatbot and has been extended
with an **agent layer**: instead of always doing one fixed retrieve-then-answer
pass, a ReAct-style planner decides step by step what to do next — search the
document store, answer from what's already been retrieved, summarize, compare
multiple documents, or reply directly from conversation memory (e.g. for a
greeting or a follow-up question that doesn't need new evidence).

The system keeps two layers of memory (short-term recent messages and a
persisted long-term summary) so multi-turn conversations stay coherent
without re-explaining context every turn.

**LLM generation runs on Groq** (fast, hosted inference — no local model
download or GPU required), embeddings run on a **configurable provider**
(a local, free, CPU-friendly HuggingFace/sentence-transformers model by
default, or OpenAI's hosted embeddings API as an alternative), vectors are
stored in **Qdrant running as its own server** (Docker container in
development/production — see `docker-compose.yml` — or any Qdrant server
reachable at `QDRANT_URL`), **Whisper** handles speech-to-text, and
Tesseract/OpenCV handle OCR on scanned PDFs and images.

> This project previously used Ollama for both LLM generation and embeddings.
> It has been fully migrated to Groq + a configurable embeddings provider;
> no Ollama installation or model pull is required or supported anymore.

---

## 2. Features

- 📄 **Multi-format ingestion** — PDF, DOCX/DOC, TXT, Markdown, JSON, Excel
  (XLSX/XLS/CSV — every sheet, with sheet-summary + row-group chunking), and
  images (JPG/PNG/TIFF/BMP/WEBP), with automatic OCR fallback for
  scanned/text-light PDFs and pure image files. Each file type is handled by
  its own loader module in `backend/loaders/`, dispatched through a single
  registry keyed by file extension.
- 🌐 **Bilingual retrieval with query expansion** — Arabic and English
  queries are normalized, translated, spell-corrected, rephrased, and
  concept-expanded (synonyms for evaluative language like "advantages" /
  "more efficient") into multiple retrieval variants so cross-language,
  typo-heavy, and semantically-phrased questions still find the right chunks.
- 🔎 **Hybrid retrieval with cross-encoder reranking** — vector similarity
  search followed by a reranking pass that blends a semantic cross-encoder
  score with lexical/bigram overlap (Arabic- and English-aware), then
  reselects for diversity (MMR-lite) and trims to a context character
  budget so irrelevant or redundant chunks don't crowd out the real answer.
- 🤖 **Agentic reasoning (ReAct loop) with semantic intent recognition** —
  the agent chooses one action at a time (`retrieve`, `generate`,
  `summarize`, `compare`, `respond`, `report`) based on the conversation so
  far. It recognises comparison/evaluation intent from natural phrasing
  ("which is better?", "pros and cons?") and resolves implicit references
  ("the two methods", "this document") from the active document and memory
  instead of always asking for clarification.
- 🧠 **Structured two-tier memory** — in-RAM short-term message history
  (summarized on message-count OR character-budget overflow) plus a
  disk-persisted long-term memory of capped, deduplicated, importance-ranked
  facts (not a single free-text paragraph) per conversation.
- 🎙️ **Voice input** — record a question; Whisper transcribes it
  (Arabic/English auto-detection with an Egyptian-Arabic-tuned second pass)
  before it's handed to the agent.
- ⚡ **Groq-backed generation** — fast hosted inference, swappable model via
  a single environment variable.
- ✂️ **Concise-by-default answers** — response length matches question
  complexity; examples are added only when they genuinely aid understanding
  or were explicitly requested, not on every answer.
- 🧩 **Pluggable embeddings** — local HuggingFace model by default, OpenAI
  embeddings as a drop-in alternative, both behind one `EMBEDDING_PROVIDER`
  switch.

---

## 3. Architecture

```
React / Next.js UI
    ↓  fetch("/api/chat")
Next.js rewrite proxy  (next.config.js → BACKEND_INTERNAL_URL,
                         e.g. http://backend:8000 in Docker,
                         http://localhost:8000 for native dev)
    ↓
FastAPI  /api/chat
    ↓
Agent (ReAct loop)  ──uses──▶  Memory Manager (short-term + summary)
    ↓ tool call
RAG Service
    ├─ retrieve   → Qdrant vector search + lexical rerank
    ├─ generate   → Groq chat completion (grounded prompt + context)
    ├─ summarize  → Groq chat completion
    ├─ compare    → Groq chat completion
    └─ respond    → Groq chat completion (memory-only)
    ↓
JSON response  { "answer": "...", "sources": "...", "stt_text": "..." }
    ↓
Frontend renders AnswerBox + SourceBox
```

The backend response shape for `/api/chat` and `/api/chat/voice` is fixed
and matched exactly by the frontend's `ChatResponse` type in
`frontend/services/api.ts`:

```json
{
  "answer": "The generated answer text",
  "sources": "Sources: file.pdf (p. 1, 2) | other.docx",
  "stt_text": ""
}
```

---

## 4. Project Structure

```
ai-doc-assistant/
├── backend/
│   ├── main.py                     FastAPI app, CORS, router mounting
│   ├── config.py                   All environment-driven settings
│   ├── requirements.txt
│   ├── .env.example
│   ├── assets/fonts/               Bundled Amiri Arabic font (PDF reports)
│   ├── loaders/                    Per-file-type document loaders (pdf/docx/text/image/excel) + registry
│   ├── agent/
│   │   ├── agent.py                ReAct loop (+ run_stream for /ws/chat)
│   │   ├── llm.py                  Groq-backed action-selection LLM (JSON mode)
│   │   ├── prompt.py                Planner system/user prompts
│   │   ├── registry.py             Tool registry factory
│   │   ├── schemas.py              Pydantic action/context schemas
│   │   ├── session.py              Per-conversation Agent registry
│   │   └── tools/                  retrieve / generate / summarize / compare / respond / report
│   ├── memory/                     Short-term + persisted long-term summary memory
│   ├── routes/
│   │   ├── chat.py                 /api/chat, /api/chat/voice, /api/chat/reset
│   │   ├── ws.py                   /ws/chat — streaming WebSocket chat
│   │   ├── upload.py               /api/upload, /api/stored-files, file download
│   │   ├── reports.py              /api/reports/generate, report download
│   │   ├── ocr.py                  /api/ocr/handwritten (see backend/HANDWRITTEN_OCR.md)
│   │   └── health.py               /api/health
│   └── services/
│       ├── rag_service.py          Core RAG pipeline (ingest, retrieve, prompt, generate)
│       ├── llm_provider.py         Groq client wrapper (invoke / chat / stream)
│       ├── embeddings_provider.py  Embeddings factory (huggingface / openai)
│       ├── db_service.py           Qdrant client/collection helpers
│       ├── storage_service.py      MinIO object storage wrapper
│       ├── report_service.py       Map-reduce summarization + PDF rendering
│       ├── audio_service.py        Whisper transcription
│       ├── ocr_service.py          Tesseract/OpenCV OCR (printed text)
│       └── handwritten_ocr_service.py  Hugging Face TrOCR (handwritten Arabic/English)
│   ├── HANDWRITTEN_OCR.md          Handwritten OCR feature docs
│   ├── Dockerfile                  Multi-stage build (venv builder → slim runtime)
│   └── .dockerignore
├── docker-compose.yml              Full stack: qdrant + minio + backend + frontend
├── .env.example                    Compose-level vars (MinIO creds, Qdrant image tag)
└── frontend/
    ├── app/                        Next.js App Router pages
    ├── components/                 ChatBox, AnswerBox, SourceBox, UploadBox, VoiceRecorder, HandwrittenOcrModal
    ├── services/api.ts             Typed fetch wrapper + WebSocket streamChat()
    ├── next.config.js              Rewrite: /api/* → BACKEND_INTERNAL_URL/api/* (standalone output)
    ├── Dockerfile                  Multi-stage build (deps → build → standalone runtime)
    └── .dockerignore
```

---

## 5. Installation

### 5a. Running with Docker Compose (recommended)

The only prerequisites are Docker and Docker Compose — everything else
(Python, Node, Qdrant, MinIO, ffmpeg, tesseract) runs inside the four
containers `docker-compose.yml` defines.

```bash
cp .env.example .env                    # Compose-level vars — MinIO creds, Qdrant image tag
cp backend/.env.example backend/.env    # fill in GROQ_API_KEY at minimum
docker compose up --build
```

> ⚠️ **Rotate your Groq API key before relying on this for anything real.**
> `backend/.env.example`'s history in this repo has, at times, shipped
> with a real-looking key checked in as the default value. `backend/.env`
> itself is git-ignored (never committed), but if you ever copied that
> key into a real deployment, generate a fresh one at
> https://console.groq.com/keys and use that instead.

This builds and starts, in dependency order (each gated on the previous
being *healthy*, not just started):

1. `qdrant` — vector store, REST on `:6333`, gRPC on `:6334`, data in a
   named volume (`qdrant_data`).
2. `minio` — object storage, API on `:9000`, console on `:9001`, data in
   `minio_data`.
3. `backend` — FastAPI, `:8000`, waits for both of the above to report
   healthy before it even starts; model weights (sentence-transformers,
   cross-encoder, Whisper) download on first boot into a `backend_model_cache`
   volume, not into the image, so they persist across `--build`s.
4. `frontend` — Next.js (standalone build), `:3000`, waits for the
   backend to report healthy.

Open `http://localhost:3000`. Tear down with `docker compose down`
(add `-v` to also delete the named volumes, i.e. wipe all indexed
documents/objects).

**Native (non-Docker) Qdrant/MinIO, Dockerized app:** point
`QDRANT_URL`/`MINIO_ENDPOINT` in `backend/.env` at wherever they're
actually running instead of `qdrant`/`minio`, and remove those two
services from `docker-compose.yml` (or just don't start them).

### 5b. Native (non-Docker) Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- `ffmpeg` and `tesseract` on your PATH (for voice input and OCR)
- A locally-running Qdrant server (`docker run -p 6333:6333 -p 6334:6334
  qdrant/qdrant:v1.19.0`, or any Qdrant instance reachable at `QDRANT_URL`)
- A [Groq API key](https://console.groq.com/keys) (free tier available)

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Open `backend/.env` and set at minimum:

```
GROQ_API_KEY=your-groq-api-key-here
```

Everything else has a working default (see [Section 6](#6-environment-variables)).

### Frontend

```bash
cd frontend
npm install
```

---

## 6. Environment Variables

All configuration lives in `backend/.env` (copy from `backend/.env.example`).
Nothing is hardcoded in source — every key below is read via `config.py`.

### LLM (Groq)

| Variable          | Default                      | Description                                                        |
|--------------------|-------------------------------|----------------------------------------------------------------------|
| `GROQ_API_KEY`      | *(required)*                 | Your Groq API key. Get one at https://console.groq.com/keys        |
| `GROQ_MODEL`        | `qwen/qwen3.8-27b`           | Model used for final answer generation, summarization, comparison, and topic-report reduction. |
| `AGENT_MODEL`       | `openai/gpt-oss-20b`         | Small/fast model for the agent's action-selection (planning) step and query rewrite/translation — deliberately a different, smaller model on a separate Groq rate-limit pool from `GROQ_MODEL`. |
| `AGENT_FALLBACK_MODEL` | `openai/gpt-oss-safeguard-20b` | Distinct-model retry target when Groq's JSON validator rejects an `AGENT_MODEL` request outright — must stay different from both `AGENT_MODEL` and `GROQ_MODEL`. |
| `LLM_TEMPERATURE`   | `0.0`                        | Sampling temperature.                                              |
| `LLM_MAX_TOKENS`    | `800`                        | Max tokens per generation.                                         |
| `LLM_TOP_P`         | `0.90`                       | Nucleus sampling parameter.                                        |

### Embeddings

| Variable              | Default                                                              | Description                                             |
|------------------------|------------------------------------------------------------------------|-----------------------------------------------------------|
| `EMBEDDING_PROVIDER`   | `huggingface`                                                         | `huggingface` (local, free, CPU) or `openai` (hosted API). |
| `EMBEDDING_MODEL`      | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`         | Model name for the selected provider.                    |
| `OPENAI_API_KEY`       | *(empty)*                                                              | Required only if `EMBEDDING_PROVIDER=openai`.            |

> ⚠️ If you change `EMBEDDING_PROVIDER` or `EMBEDDING_MODEL` after documents
> have already been ingested, the existing Qdrant collection's vector
> dimension won't match anymore. `ensure_collection()` in
> `backend/services/db_service.py` detects this on startup and raises a
> clear error rather than silently deleting/recreating the collection —
> either delete the collection yourself (Qdrant's REST API or the
> `qdrant_data` volume) or point `QDRANT_COLLECTION` at a fresh name, then
> re-upload. Vectors from different embedding models are never compatible.

### Vector store / RAG / Audio / Agent / Memory

See `backend/.env.example` for the full list (Qdrant URL/collection,
chunking, retrieval `k`, OCR toggle, Whisper model size, agent iteration
limit, memory window sizes, etc.) — all have sensible defaults and rarely
need to change. Notable additions:

| Variable                   | Default                                        | Description                                                                 |
|-----------------------------|-------------------------------------------------|-------------------------------------------------------------------------------|
| `QDRANT_URL`                | `http://localhost:6333`                        | Qdrant server address. `docker-compose.yml` overrides this to `http://qdrant:6333` for the backend container. |
| `QDRANT_CONNECT_RETRIES`    | `5`                                             | Bounded retry attempts for Qdrant operations (startup attach, collection checks, deletes). |
| `QDRANT_RETRY_DELAY_SECONDS`| `2`                                             | Delay between retry attempts.                                              |
| `FRONTEND_ORIGIN`           | `http://localhost:3000`                        | Comma-separated CORS allowlist — the frontend origin(s) allowed to call the API with credentials. |
| `QUERY_EXPANSION_ENABLED`   | `true`                                          | Adds same-language synonym/concept query variants for semantic questions.  |
| `RERANK_USE_CROSS_ENCODER`  | `true`                                          | Blend a semantic cross-encoder score into reranking (falls back to lexical-only if the model can't load). |
| `CROSS_ENCODER_MODEL`       | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`    | Multilingual, CPU-friendly, no API key.                                    |
| `RERANK_ALPHA`              | `0.6`                                           | Weight of the cross-encoder score vs. the lexical score (0-1).             |
| `RERANK_DIVERSIFY`          | `true`                                          | Reselect the final chunk set for diversity (MMR-lite) after scoring.       |
| `MMR_LAMBDA`                | `0.7`                                           | Relevance vs. diversity trade-off for `RERANK_DIVERSIFY` (0-1).            |
| `MAX_CONTEXT_CHARS`         | `6000`                                          | Character budget for retrieved context sent to the LLM per answer.        |
| `MEMORY_MAX_FACTS`          | `40`                                            | Cap on stored long-term-memory facts (lowest importance/oldest evicted first). |
| `MEMORY_SUMMARY_MAX_CHARS`  | `1200`                                          | Character budget for rendered long-term-memory facts per prompt.          |
| `MEMORY_MAX_CHARS`          | `12000`                                         | Short-term memory also summarizes once buffered message text crosses this budget. |

---

## 7. Configuring the Groq API

1. Create a free account at https://console.groq.com.
2. Generate an API key under **API Keys**.
3. Set `GROQ_API_KEY` in `backend/.env`.
4. (Optional) Change `GROQ_MODEL` / `AGENT_MODEL` to any model your account
   has access to — check https://console.groq.com/docs/models for the
   current list. A smaller/faster model for `AGENT_MODEL` (the planner)
   can noticeably reduce per-turn latency since it may be called multiple
   times per question.

No other code changes are needed — every Groq call in the backend goes
through the single `backend/services/llm_provider.py` module.

---

## 8. Configuring the Embedding Provider

**Default — local, free, no API key (`EMBEDDING_PROVIDER=huggingface`):**

Uses `sentence-transformers` to run a small multilingual model on CPU. The
first run downloads the model (a few hundred MB) from HuggingFace; after
that it's fully offline.

```
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

**Alternative — OpenAI hosted embeddings (`EMBEDDING_PROVIDER=openai`):**

```
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=your-openai-api-key-here
```

Both providers return LangChain-compatible embeddings objects, so Qdrant
storage and retrieval logic in `rag_service.py` and `db_service.py` are
completely unaffected by which one you choose.

---

## 9. Running the Backend

> Using Docker Compose (`docker compose up --build`)? This and the next
> section don't apply — the backend/frontend are already running as
> containers. This is the native (non-Docker) path, requiring a Qdrant
> server already reachable at `QDRANT_URL` (see [5b](#5b-native-non-docker-installation)).

```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API is now available at `http://localhost:8000/api/*`. Health check:

```bash
curl http://localhost:8000/api/health
# {"status": "ok"}
```

## 10. Running the Frontend

```bash
cd frontend
npm run dev
```

Open `http://localhost:3000`. In development, `next.config.js` rewrites
every `/api/*` request to `http://localhost:8000/api/*`, so the backend
must be running on port 8000 (or update `next.config.js` if you use a
different port/host).

For production, either keep the rewrite pointed at your deployed backend
URL, or serve the frontend and backend behind a shared reverse proxy.

---

## 11. Building the Vector Database / Uploading Documents

There is no separate "build the database" step — uploading a document
ingests it directly:

1. With both servers running, open the app in your browser.
2. Use the **Documents** panel in the sidebar (or `POST /api/upload`
   directly) to upload one or more PDF/DOCX/TXT/MD/JSON/XLSX/XLS/CSV/image
   files.
3. The backend dispatches each file to its loader (`backend/loaders/`),
   OCRs it if needed (scanned PDFs / images), splits it into chunks
   (Excel sheets get sheet-summary + row-group chunks instead of generic
   text splitting), embeds the chunks with the configured embeddings
   provider, and stores the vectors in the local Qdrant collection.
4. Already-processed files (tracked by content hash in
   `processed_files.json`) are skipped on re-upload.

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "files=@/path/to/document.pdf"
```

List everything that's been ingested:

```bash
curl http://localhost:8000/api/stored-files
```

---

## 12. Chat Usage

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What does the document say about pricing?", "language": "auto"}'
```

Response:

```json
{
  "answer": "...",
  "sources": "Sources: pricing.pdf (p. 2, 3)",
  "stt_text": ""
}
```

Voice input: `POST /api/chat/voice` with a multipart `audio` file field
(the frontend's microphone button does this automatically).

Reset a conversation's memory: `POST /api/chat/reset`.

In the UI, use the language toggle (Auto / العربية / English) above the
input box to bias language detection, or just type — the backend
auto-detects Arabic vs. English per message.

---

## 13. Troubleshooting

**"GROQ_API_KEY is not set" error on first chat request**
Set `GROQ_API_KEY` in `backend/.env` and restart the backend. Keys are
read once at process start via `python-dotenv`.

**Frontend shows "Internal Server Error" with no further detail**
This means the response never reached FastAPI's own JSON error handler —
most commonly a stale/crashed backend process, a port mismatch between
`next.config.js` and where `uvicorn` is actually listening, or a request
that exceeded the client-side timeout in `frontend/services/api.ts`
(default 60s). Check the terminal running `uvicorn` for the actual
traceback, and check the browser Network tab for the real HTTP status
code and response body of the failed `/api/chat` call. As of this version,
every FastAPI error response includes a real `detail` message and the
frontend surfaces it directly instead of a generic string — a bare
"Internal Server Error" now specifically indicates the error happened
*before* FastAPI, not inside it.

**Chat is slow**
Each `/api/chat` call can trigger several Groq calls in sequence (query
translation/rephrasing/spelling variants during retrieval, the agent's
planning step for each ReAct iteration, and the final answer generation).
Groq is fast per call, but the cumulative latency is still visible on
first use. Reducing `AGENT_MAX_ITERATIONS` or using a smaller/faster
`AGENT_MODEL` for planning will reduce this further.

**Embeddings dimension mismatch / Qdrant errors after switching providers**
Different embedding models produce vectors of different sizes.
`ensure_collection()` (`backend/services/db_service.py`) checks this on
startup and raises a clear `RuntimeError` naming both the found and
expected schema instead of silently deleting/recreating anything — delete
the Qdrant collection yourself (or point `QDRANT_COLLECTION` at a new
name) and re-upload your documents.

**`GET /api/health` reports `"qdrant": "unreachable"`**
Under Docker Compose: `docker compose ps` — is the `qdrant` container
healthy? `docker compose logs qdrant`. The backend's `depends_on:
condition: service_healthy` should keep it from starting before Qdrant is
up, but if Qdrant crashes *after* the backend started, retrieval/chat
will fail until it's back — the backend retries (`QDRANT_CONNECT_RETRIES`
/ `QDRANT_RETRY_DELAY_SECONDS`) rather than crashing, so restarting the
`qdrant` container alone (`docker compose restart qdrant`) is usually
enough to recover, no backend restart needed.
Native (non-Docker): confirm `QDRANT_URL` in `backend/.env` actually
points at a running Qdrant server.

**"No relevant documents were found" for everything**
Confirm at least one file was successfully uploaded (`GET
/api/stored-files`) and that the Qdrant collection actually has vectors —
check the backend startup log for `Loaded existing DB — N vectors`.

**Voice input fails**
Ensure `ffmpeg` and `tesseract` are installed and on your PATH, or set
`FFMPEG_PATH` / `TESSERACT_CMD` explicitly in `.env`. Under Docker Compose
these are already installed in the backend image and set via
`docker-compose.yml`'s `environment:` block — this only applies to native
(non-Docker) setups.

**CORS errors when calling the backend directly (bypassing the Next.js
proxy)**
`main.py` restricts origins to `FRONTEND_ORIGIN` (default
`http://localhost:3000`) — a comma-separated allowlist, not a wildcard
(browsers reject `allow_origins=["*"]` combined with
`allow_credentials=True` anyway). If you're calling the API from a
different origin, add it to `FRONTEND_ORIGIN` in `backend/.env`.

---

## 14. Migration Notes (Ollama → Groq)

This project no longer uses Ollama in any form:

- `langchain-ollama` has been removed from `requirements.txt`.
- `backend/services/rag_service.py` and `backend/agent/llm.py` now use
  `backend/services/llm_provider.py` (Groq) instead of `OllamaLLM`.
- Embeddings now go through `backend/services/embeddings_provider.py`
  instead of `OllamaEmbeddings`.
- `OLLAMA_LLM_MODEL` / `OLLAMA_EMBED_MODEL` env vars have been replaced by
  `GROQ_API_KEY` / `GROQ_MODEL` / `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL`.
- The RAG pipeline itself (document loading → chunking → embeddings →
  Qdrant → retrieval → prompt construction → LLM generation) is
  unchanged — only the model providers were swapped.
