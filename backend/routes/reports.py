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
    # Optional: the whole-document report path (no `topic`) still reads a
    # named file directly from MinIO/the stored-files registry, as before,
    # so `filename` is required in that case. A topic-scoped report (see
    # `topic` below) searches the caller's ENTIRE uploaded knowledge base
    # via semantic retrieval and does NOT require a filename — `filename`
    # becomes an optional scope-down to one specific document when both
    # are given together, matching the chat-driven "report" tool's
    # (document, topic) resolution in agent/tools/report_tool.py.
    filename: Optional[str] = None
    # Scope the report to a subject/topic instead of summarizing a whole
    # document (see report_service.generate_topic_report). This is the
    # primary way to request "a report about X" without knowing/naming any
    # uploaded filename — semantic retrieval finds the relevant material
    # across every document in the conversation's knowledge base.
    topic: Optional[str] = None
    # Required whenever `topic` is set — a topic-scoped report retrieves
    # via the vector store (see Document Isolation) and must be scoped to
    # one conversation's own documents, same as any other retrieval. The
    # whole-document path (no topic) reads the named file directly and
    # doesn't need it.
    conversation_id: Optional[str] = None


@router.post("/reports/generate")
async def generate_report(request: ReportRequest):
    """
    Generate a comprehensive PDF report.

    - `topic` given, no `filename`: searches the caller's entire uploaded
      knowledge base for that topic via semantic retrieval and builds the
      report from whatever relevant material is found across ALL
      documents — no filename required (see report_service.generate_topic_report).
    - `topic` AND `filename` given: same topic-scoped retrieval, narrowed
      to that one document.
    - `filename` only (no `topic`): summarizes that whole document, looked
      up by original filename (as returned by GET /api/stored-files) —
      unchanged from before.

    Stores the PDF in MinIO and returns a presigned download URL alongside
    the backend-proxied download path.
    """
    has_topic = bool(request.topic and request.topic.strip())
    has_filename = bool(request.filename and request.filename.strip())

    if not has_topic and not has_filename:
        raise HTTPException(
            status_code=400,
            detail="Either 'topic' or 'filename' is required.",
        )
    if has_topic and not (request.conversation_id and request.conversation_id.strip()):
        raise HTTPException(
            status_code=400,
            detail="conversation_id is required when a topic is given (topic reports retrieve "
            "from the vector store, which is scoped per conversation).",
        )

    try:
        if has_topic:
            result = report_service.generate_topic_report(
                request.topic, request.conversation_id,
                document=request.filename if has_filename else None,
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

    subject = request.topic if has_topic else request.filename
    return {
        "message": f"Report generated for '{subject}'.",
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
