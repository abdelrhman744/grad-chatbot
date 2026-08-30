"""
report_tool.py

Terminal tool: generates a comprehensive PDF summary report for an
uploaded document and hands back a downloadable link, entirely inside
the chat — no Swagger/manual API calls needed.

Target-document resolution order:
  1. an explicit filename the planner extracted from the user's message
     (action.arguments.document)
  2. the conversation's "active document" (the file most recently
     retrieved from, or most recently reported on, in this conversation)
  3. the only uploaded document, if there's exactly one
  4. otherwise: ask the user which document they mean (this tool sets
     `context.needs_clarification` and returns a question as the answer,
     instead of a report — the next turn's message naming a file will be
     resolved back here by the planner/active-document logic)
"""

from __future__ import annotations

import logging
from difflib import get_close_matches
from typing import Callable, Optional

from agent.schemas import ExecutionContext
from services import rag_service, report_service

log = logging.getLogger("agent.tools.report")


class ReportTool:
    name = "report"

    def __init__(
        self,
        conversation_id: str,
        active_document_provider: Optional[Callable[[], Optional[str]]] = None,
        active_document_setter: Optional[Callable[[str], None]] = None,
    ):
        # Used only by the topic-scoped report path (_run_topic_report),
        # which retrieves via rag_service.retrieve() and must therefore be
        # scoped to this conversation's own documents like any other
        # retrieval — see Document Isolation. The whole-document report
        # path (run(), below) reads a named file directly from MinIO/the
        # global registry and is NOT scoped by conversation_id — see the
        # Document Isolation implementation notes for why that's a
        # deliberate, documented limitation rather than an oversight.
        self.conversation_id = conversation_id
        self._get_active = active_document_provider or (lambda: None)
        self._set_active = active_document_setter or (lambda _f: None)

    def run(
        self, context: ExecutionContext, document: Optional[str] = None, topic: Optional[str] = None
    ) -> ExecutionContext:
        is_ar = context.language == "ar"
        topic = (topic or "").strip()

        if topic:
            return self._run_topic_report(context, topic=topic, document=document, is_ar=is_ar)

        try:
            files = rag_service.list_stored_files()
        except Exception as e:
            log.exception("Could not list stored files for report tool")
            context.answer = (
                f"تعذّر الوصول إلى الملفات المخزّنة: {e}" if is_ar
                else f"Could not access stored files: {e}"
            )
            return context

        if not files:
            context.answer = (
                "لا توجد مستندات مرفوعة بعد لإنشاء تقرير عنها. ارفع مستندًا أولاً."
                if is_ar else
                "No documents have been uploaded yet to generate a report from. "
                "Please upload a document first."
            )
            return context

        filenames = [f["filename"] for f in files]
        target, ambiguous_candidates = self._resolve_target(filenames, document)

        if target is None:
            if ambiguous_candidates is not None:
                # Multiple documents, no way to tell which one was meant.
                context.needs_clarification = ambiguous_candidates
                listing = "\n".join(f"- {name}" for name in ambiguous_candidates)
                context.answer = (
                    f"عندك أكتر من مستند مرفوع. عايز التقرير عن أنهي ملف؟\n\n{listing}"
                    if is_ar else
                    f"You have more than one uploaded document. Which one would you like "
                    f"the report for?\n\n{listing}"
                )
            else:
                # A document name was given but didn't match anything.
                listing = "\n".join(f"- {name}" for name in filenames)
                context.answer = (
                    f"معنديش ملف اسمه '{document}'. الملفات المتاحة:\n\n{listing}"
                    if is_ar else
                    f"I couldn't find a document named '{document}'. Available files:\n\n{listing}"
                )
            return context

        try:
            result = report_service.generate_report(target)
        except FileNotFoundError as e:
            context.answer = str(e)
            return context
        except ValueError as e:
            context.answer = (
                f"تعذّر إنشاء تقرير عن '{target}': {e}" if is_ar
                else f"Could not generate a report for '{target}': {e}"
            )
            return context
        except Exception as e:
            log.exception("Report generation failed inside agent tool")
            context.answer = (
                f"حصل خطأ أثناء إنشاء التقرير: {e}" if is_ar
                else f"An error occurred while generating the report: {e}"
            )
            return context

        self._set_active(target)

        context.report = {
            "filename": target,
            "object_name": result["object_name"],
            "download_url": result["download_url"],
            "proxy_download_path": f"/api/reports/{result['object_name']}/download",
        }
        context.answer = (
            f"تم إنشاء تقرير احترافي شامل عن '{target}' بنجاح. تقدر تنزّله من الرابط أدناه."
            if is_ar else
            f"A comprehensive professional report for '{target}' has been generated "
            f"successfully. You can download it below."
        )
        context.observations.append({"tool": "report", "status": "generated", "document": target})

        return context

    def _run_topic_report(
        self, context: ExecutionContext, topic: str, document: Optional[str], is_ar: bool
    ) -> ExecutionContext:
        """
        Topic-scoped report path: searches all uploaded knowledge for
        `topic` (or, if `document` was also named, within that one document
        only) instead of summarizing one whole document end to end.
        """
        resolved_document: Optional[str] = None

        if document:
            try:
                filenames = [f["filename"] for f in rag_service.list_stored_files()]
            except Exception as e:
                log.exception("Could not list stored files for topic report tool")
                context.answer = (
                    f"تعذّر الوصول إلى الملفات المخزّنة: {e}" if is_ar
                    else f"Could not access stored files: {e}"
                )
                return context

            resolved_document = self._fuzzy_match(document, filenames)
            if resolved_document is None:
                listing = "\n".join(f"- {name}" for name in filenames)
                context.answer = (
                    f"معنديش ملف اسمه '{document}'. الملفات المتاحة:\n\n{listing}"
                    if is_ar else
                    f"I couldn't find a document named '{document}'. Available files:\n\n{listing}"
                )
                return context

        try:
            result = report_service.generate_topic_report(
                topic, self.conversation_id, document=resolved_document
            )
        except ValueError as e:
            # Includes the "not enough information about this topic" case —
            # surfaced as-is instead of generating a hallucinated report.
            context.answer = str(e)
            return context
        except Exception as e:
            log.exception("Topic report generation failed inside agent tool")
            context.answer = (
                f"حصل خطأ أثناء إنشاء التقرير عن '{topic}': {e}" if is_ar
                else f"An error occurred while generating the report about '{topic}': {e}"
            )
            return context

        if resolved_document:
            self._set_active(resolved_document)

        context.report = {
            "filename": topic,
            "object_name": result["object_name"],
            "download_url": result["download_url"],
            "proxy_download_path": f"/api/reports/{result['object_name']}/download",
        }
        context.answer = (
            f"تم إنشاء تقرير احترافي شامل عن '{topic}' بنجاح. تقدر تنزّله من الرابط أدناه."
            if is_ar else
            f"A comprehensive professional report about '{topic}' has been generated "
            f"successfully. You can download it below."
        )
        context.observations.append({"tool": "report", "status": "generated", "topic": topic})

        return context

    # ── Internal ─────────────────────────────────────────────────────────

    def _resolve_target(
        self, filenames: list[str], requested: Optional[str]
    ) -> tuple[Optional[str], Optional[list[str]]]:
        """
        Returns (target_filename, None) on success, or
        (None, ambiguous_candidates) when the caller should ask the user
        to pick, or (None, None) when a requested name didn't match
        anything at all.
        """
        if requested:
            match = self._fuzzy_match(requested, filenames)
            if match:
                return match, None
            return None, None  # named a file, but no match — not ambiguous, just not found

        active = self._get_active()
        if active and active in filenames:
            return active, None

        if len(filenames) == 1:
            return filenames[0], None

        return None, filenames  # ambiguous: multiple files, no hint

    @staticmethod
    def _fuzzy_match(requested: str, filenames: list[str]) -> Optional[str]:
        requested_norm = requested.strip().lower()

        # Exact / substring match first (handles "the networks pdf" style refs).
        for name in filenames:
            if requested_norm == name.lower():
                return name
        for name in filenames:
            stem = name.lower().rsplit(".", 1)[0]
            if requested_norm in name.lower() or requested_norm in stem or stem in requested_norm:
                return name

        # Fall back to fuzzy string matching for typos/partial names.
        close = get_close_matches(requested, filenames, n=1, cutoff=0.6)
        return close[0] if close else None
