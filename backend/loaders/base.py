"""Shared helpers used by every loader module."""

import time


def make_meta(filename: str, file_type: str, **extra) -> dict:
    """Build the standard per-document metadata dict every loader returns.
    `page` defaults to 0 (non-paged formats); loaders that have real page/
    sheet numbers overwrite it. `**extra` lets a loader attach type-specific
    fields (e.g. Excel's `sheet_name`) without duplicating the base keys."""
    return {
        "source": filename,
        "file_type": file_type,
        "page": 0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **extra,
    }


def clean_text(text) -> str:
    return str(text).strip() if text is not None else ""
