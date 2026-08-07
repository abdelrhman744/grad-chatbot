from fastapi import APIRouter

from services import db_service, storage_service

router = APIRouter()


@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "minio": "ok" if storage_service.is_available() else "unreachable",
        "qdrant": "ok" if db_service.is_available() else "unreachable",
    }
