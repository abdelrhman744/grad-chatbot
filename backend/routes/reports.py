import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services import report_service

log = logging.getLogger("routes.reports")

router = APIRouter()


class ReportRequest(BaseModel):
    filename: str


@router.post("/reports/generate")
async def generate_report(request: ReportRequest):
    """
    Generate a comprehensive PDF summary report for a single previously
    uploaded document (looked up by its original filename, as returned by
    GET /api/stored-files). Stores the PDF in MinIO and returns a
    presigned download URL alongside the backend-proxied download path.
    """
    if not request.filename.strip():
        raise HTTPException(status_code=400, detail="filename is required.")

    try:
        result = report_service.generate_report(request.filename)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
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
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Report not found: {e}")

    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{object_name}"'},
    )
