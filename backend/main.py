import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routes.upload import router as upload_router
from routes.chat import router as chat_router
from routes.health import router as health_router
from routes.ws import router as ws_router
from routes.reports import router as reports_router
from services.rag_service import load_existing_db

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

app = FastAPI(title="AI Document Assistant", version="2.0.0")

# allow_origins=["*"] combined with allow_credentials=True is rejected by
# browsers (and a broad hole regardless) — FRONTEND_ORIGIN is a real,
# env-driven allowlist instead (see config.py).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.FRONTEND_ORIGIN.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(health_router, prefix="/api")
app.include_router(reports_router, prefix="/api")
app.include_router(ws_router)  # exposes /ws/chat (no /api prefix, plain WebSocket path)


@app.on_event("startup")
async def on_startup():
    """Attach to any existing Qdrant collection so retrieval works
    immediately after a restart, without waiting for a new upload."""
    load_existing_db()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
