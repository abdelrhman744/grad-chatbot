# DocAssist AI (merged)

Frontend from **grad-chatbot-main** + RAG / OCR / vector_db from **project-copy**.

## Structure (unchanged from project-copy)

```
docassist/
├── frontend/          # Next.js UI (grad-chatbot)
├── ocr/
│   ├── data/          # incoming PDFs + metadata JSON
│   ├── processed/     # after OCR
│   ├── output/        # text.txt + metadata.json per doc
│   └── ...
├── vector_db/
│   ├── xlsx/          # incoming Excel folders (file + json)
│   ├── done_docs/     # after embedding (OCR path)
│   ├── done_xlsx/     # after embedding (Excel path)
│   ├── qdrant_db/
│   ├── whoosh_index/
│   ├── model_manager.py   # embedding model loaded ONCE
│   └── ...
├── rag/               # agent, hybrid search, reranker, memory
├── server.py          # FastAPI: /api/chat, /api/upload, /api/upload/xlsx
└── .env               # GROQ_API_KEY, etc.
```

## Upload rules

- **PDF / Docs**: select the document **and** its metadata `.json` with the **same base name** (e.g. `lecture.pdf` + `lecture.json`).
- **Excel**: select the `.xlsx` **and** its metadata `.json` with the **same base name**.

## Embedding model

Loaded **once** at server startup via `vector_db.model_manager` (shared with `rag.search`). Uploads never reload the model.

## Run

```bash
# Backend
cd docassist
cp .env.example .env   # set GROQ_API_KEY
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000

# Frontend (another terminal)
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 — Next.js proxies `/api/*` to the backend.
