import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services import report_service
from services.storage_service import StorageUnavailableError

log = logging.getLogger("routes.reports")

router = APIRouter()


class ReportRequest(BaseModel):
    filename: str
    # Optional: scope the report to a subject/topic instead of summarizing
    # the whole document (see report_service.generate_topic_report). Kept
    # here for parity with the chat-driven "report" tool; the frontend
    # currently only reaches this feature through natural-language chat.
    topic: Optional[str] = None
    # Required only when `topic` is set — a topic-scoped report retrieves
    # via the vector store (see Document Isolation) and must be scoped to
    # one conversation's own documents, same as any other retrieval. The
    # whole-document path (no topic) reads the named file directly and
    # doesn't need it.
    conversation_id: Optional[str] = None


@router.post("/reports/generate")
async def generate_report(request: ReportRequest):
    """
    Generate a comprehensive PDF report, looked up by original filename (as
    returned by GET /api/stored-files). Without `topic`, summarizes the
    whole document. With `topic`, scopes the report to that subject within
    the named document instead. Stores the PDF in MinIO and returns a
    presigned download URL alongside the backend-proxied download path.
    """
    if not request.filename.strip():
        raise HTTPException(status_code=400, detail="filename is required.")

    has_topic = bool(request.topic and request.topic.strip())
    if has_topic and not (request.conversation_id and request.conversation_id.strip()):
        raise HTTPException(
            status_code=400,
            detail="conversation_id is required when a topic is given (topic reports retrieve "
            "from the vector store, which is scoped per conversation).",
        )

    try:
        if has_topic:
            result = report_service.generate_topic_report(
                request.topic, request.conversation_id, document=request.filename
            )
        else:
            result = report_service.generate_report(request.filename)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except StorageUnavailableError as e:
        raise HTTPException(status_code=503, detail=f"Report storage is currently unavailable: {e}")
    except Exception as e:
        log.exception("Report generation failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    return {
        "message": f"Report generated for '{request.filename}'.",
        "object_name": result["object_name"],
        "download_url": result["download_url"],
        "proxy_download_path": f"/api/reports/{result['object_name']}/download",
        "language": result["language"],
    }


@router.get("/reports/{object_name}/download")
async def download_report(object_name: str):
    """
    Stream a previously generated report's PDF bytes back through the
    backend. Used as a fallback when a direct presigned MinIO URL isn't
    reachable from the client (e.g. MinIO bound to an internal hostname).
    """
    try:
        pdf_bytes = report_service.get_report_bytes(object_name)
    except StorageUnavailableError as e:
        raise HTTPException(status_code=503, detail=f"Report storage is currently unavailable: {e}")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Report not found: {e}")

    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{object_name}"'},
    )
