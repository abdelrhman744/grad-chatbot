# AI Document Assistant

An **agentic, bilingual (Arabic/English) Retrieval-Augmented Generation system**
for chatting with your own documents — LLM generation via the **Groq API**,
local **Qdrant** vector storage, and a configurable embeddings provider.

Upload PDFs, Word docs, text, JSON, or scanned images; ask questions by typing
or by voice; get grounded, cited answers from an autonomous agent that decides
for itself when to search your documents, when to rely on conversation memory,
and when to summarize or compare what it has found.

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

**Setup for these three:**

```bash
# 1. Start MinIO (uploads + generated reports bucket)
docker compose up -d
# Console: http://localhost:9001  (minioadmin / minioadmin)

# 2. Install the new backend deps (already in requirements.txt)
pip install -r backend/requirements.txt

# 3. Copy env files and adjust if needed
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local   # optional, only if backend isn't on :8000
```

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
stored in a local/embedded **Qdrant** instance, **Whisper** handles
speech-to-text, and Tesseract/OpenCV handle OCR on scanned PDFs and images.

> This project previously used Ollama for both LLM generation and embeddings.
> It has been fully migrated to Groq + a configurable embeddings provider;
> no Ollama installation or model pull is required or supported anymore.

---

## 2. Features

- 📄 **Multi-format ingestion** — PDF, DOCX/DOC, TXT, Markdown, JSON, and
  images (JPG/PNG/TIFF/BMP/WEBP), with automatic OCR fallback for
  scanned/text-light PDFs and pure image files.
- 🌐 **Bilingual retrieval** — Arabic and English queries are normalized,
  translated, spell-corrected, and rephrased into multiple retrieval
  variants so cross-language and typo-heavy questions still find the right
  chunks.
- 🔎 **Hybrid retrieval** — vector similarity search followed by a lexical
  reranking pass (keyword + bigram overlap) tuned for Arabic and English.
- 🤖 **Agentic reasoning (ReAct loop)** — the agent chooses one action at a
  time (`retrieve`, `generate`, `summarize`, `compare`, `respond`) based on
  the conversation so far, instead of following one fixed pipeline.
- 🧠 **Two-tier memory** — in-RAM short-term message history plus a
  disk-persisted, LLM-maintained long-term summary per conversation.
- 🎙️ **Voice input** — record a question; Whisper transcribes it
  (Arabic/English auto-detection with an Egyptian-Arabic-tuned second pass)
  before it's handed to the agent.
- ⚡ **Groq-backed generation** — fast hosted inference, swappable model via
  a single environment variable.
- 🧩 **Pluggable embeddings** — local HuggingFace model by default, OpenAI
  embeddings as a drop-in alternative, both behind one `EMBEDDING_PROVIDER`
  switch.

---

## 3. Architecture

```
React / Next.js UI
    ↓  fetch("/api/chat")
Next.js rewrite proxy  (next.config.js → http://localhost:8000)
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
│   │   └── health.py               /api/health
│   └── services/
│       ├── rag_service.py          Core RAG pipeline (ingest, retrieve, prompt, generate)
│       ├── llm_provider.py         Groq client wrapper (invoke / chat / stream)
│       ├── embeddings_provider.py  Embeddings factory (huggingface / openai)
│       ├── db_service.py           Qdrant client/collection helpers
│       ├── storage_service.py      MinIO object storage wrapper
│       ├── report_service.py       Map-reduce summarization + PDF rendering
│       ├── audio_service.py        Whisper transcription
│       └── ocr_service.py          Tesseract/OpenCV OCR
├── docker-compose.yml              Local MinIO instance
└── frontend/
    ├── app/                        Next.js App Router pages
    ├── components/                 ChatBox, AnswerBox, SourceBox, UploadBox, VoiceRecorder
    ├── services/api.ts             Typed fetch wrapper + WebSocket streamChat()
    └── next.config.js              Dev-time rewrite: /api/* → http://localhost:8000/api/*
```

---

## 5. Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- `ffmpeg` and `tesseract` on your PATH (for voice input and OCR)
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
| `GROQ_MODEL`        | `llama-3.3-70b-versatile`    | Model used for answer generation, translation, summarization, etc. |
| `AGENT_MODEL`       | same as `GROQ_MODEL`         | Model used for the agent's action-selection (planning) step.       |
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
> have already been ingested, delete the Qdrant collection (or point
> `QDRANT_PATH`/`QDRANT_COLLECTION` at a fresh path) and re-upload — vectors
> from different embedding models are not compatible with each other.

### Vector store / RAG / Audio / Agent / Memory

See `backend/.env.example` for the full list (Qdrant path/collection,
chunking, retrieval `k`, OCR toggle, Whisper model size, agent iteration
limit, memory window sizes, etc.) — all have sensible defaults and rarely
need to change.

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
   directly) to upload one or more PDF/DOCX/TXT/MD/JSON/image files.
3. The backend loads each file, OCRs it if needed (scanned PDFs / images),
   splits it into chunks, embeds the chunks with the configured embeddings
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
Different embedding models produce vectors of different sizes. If you
change `EMBEDDING_PROVIDER` or `EMBEDDING_MODEL` after documents are
already indexed, delete `QDRANT_PATH` (or point it at a new empty
directory) and re-upload your documents.

**"No relevant documents were found" for everything**
Confirm at least one file was successfully uploaded (`GET
/api/stored-files`) and that the Qdrant collection actually has vectors —
check the backend startup log for `Loaded existing DB — N vectors`.

**Voice input fails**
Ensure `ffmpeg` and `tesseract` are installed and on your PATH, or set
`FFMPEG_PATH` / `TESSERACT_CMD` explicitly in `.env`.

**CORS errors when calling the backend directly (bypassing the Next.js
proxy)**
`main.py` allows all origins by default (`allow_origins=["*"]`) — if
you've changed this for production, add your frontend's origin explicitly.

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
