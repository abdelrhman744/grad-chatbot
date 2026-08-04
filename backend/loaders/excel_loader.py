"""
Excel/CSV loading (.xlsx, .xls, .csv).

Reads every sheet with pandas and turns each one into:
  1. one "sheet summary" chunk (columns, row count, a few sample rows), and
  2. multiple "row group" chunks (batches of rows serialized as
     "Column: Value | ..." text, sized to roughly match settings.CHUNK_SIZE).

Unlike PDF/DOCX, no temp file is needed — pandas reads directly from the
in-memory bytes. The chunks this module returns are final and deliberately
sized; `rag_service.update_db_files()` must NOT re-run the text splitter
over them (see the file_type == "excel" bypass there).
"""

import io
import logging
import os
from typing import List

import pandas as pd
from langchain_core.documents import Document

from config import settings
from .base import make_meta

log = logging.getLogger("loaders.excel")


def _read_sheets(filename: str, data: bytes) -> "dict[str, pd.DataFrame]":
    ext = filename.lower().rsplit(".", 1)[-1]
    buf = io.BytesIO(data)

    if ext == "csv":
        df = pd.read_csv(buf, sep=None, engine="python")
        name = os.path.splitext(os.path.basename(filename))[0] or "Sheet1"
        return {name: df}

    # .xlsx requires openpyxl; .xls (legacy binary format) requires xlrd —
    # xlrd>=2.0 dropped .xlsx support entirely, so this split is mandatory.
    engine = "openpyxl" if ext == "xlsx" else "xlrd"
    sheets = pd.read_excel(buf, sheet_name=None, engine=engine)
    return sheets


def _clean_columns(cols) -> list:
    out = []
    for i, c in enumerate(cols):
        s = str(c).strip()
        out.append(s if s and not s.startswith("Unnamed") else f"Column{i + 1}")
    return out


def _clean_cell(value):
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d") if value.time() == value.time().min else str(value)
    return str(value).strip()


def _serialize_row(df: "pd.DataFrame", pos: int) -> str:
    row = df.iloc[pos]
    parts = []
    for col in df.columns:
        v = _clean_cell(row[col])
        if v:
            parts.append(f"{col}: {v}")
    return f"Row {pos + 2}: " + " | ".join(parts)  # +2 = 1 header row + 1-based row numbering


def _rows_per_group(df: "pd.DataFrame") -> "tuple[int, int]":
    sample_n = min(5, len(df))
    sample = [_serialize_row(df, i) for i in range(sample_n)] or [""]
    avg_len = max(1, sum(len(s) for s in sample) // len(sample))

    rows_per_group = max(
        settings.EXCEL_ROWS_PER_CHUNK_MIN,
        min(settings.EXCEL_ROWS_PER_CHUNK_MAX, settings.CHUNK_SIZE // avg_len or 1),
    )
    overlap_ratio = (settings.CHUNK_OVERLAP / settings.CHUNK_SIZE) if settings.CHUNK_SIZE else 0
    overlap_rows = min(rows_per_group - 1, round(rows_per_group * overlap_ratio))
    return rows_per_group, max(0, overlap_rows)


def _build_summary_chunk(sheet_name: str, df: "pd.DataFrame", filename: str,
                          sheet_index: int, total_sheets: int, truncated_from: int | None) -> Document:
    lines = [
        f"[Sheet: {sheet_name} — Summary]",
        f"This sheet has {len(df)} rows and {len(df.columns)} columns: {', '.join(map(str, df.columns))}.",
    ]
    if truncated_from:
        lines.append(f"(Showing first {len(df)} of {truncated_from} rows — sheet was truncated for indexing.)")

    sample_n = min(settings.EXCEL_SUMMARY_SAMPLE_ROWS, len(df))
    if sample_n:
        lines.append("Sample data:")
        lines.extend(_serialize_row(df, p) for p in range(sample_n))

    return Document(
        page_content="\n".join(lines),
        metadata=make_meta(
            filename, "excel", page=sheet_index, sheet_name=sheet_name,
            chunk_type="sheet_summary", row_range=None,
            total_sheets=total_sheets, sheet_row_count=len(df),
        ),
    )


def _build_row_group_chunks(sheet_name: str, df: "pd.DataFrame", filename: str,
                             sheet_index: int, total_sheets: int) -> List[Document]:
    docs: List[Document] = []
    n = len(df)
    if n == 0:
        return docs

    rows_per_group, overlap_rows = _rows_per_group(df)
    step = max(1, rows_per_group - overlap_rows)

    i = 0
    while i < n:
        end = min(i + rows_per_group, n)
        text = f"[Sheet: {sheet_name}]\n" + "\n".join(_serialize_row(df, p) for p in range(i, end))
        docs.append(Document(
            page_content=text,
            metadata=make_meta(
                filename, "excel", page=sheet_index, sheet_name=sheet_name,
                chunk_type="row_group", row_range=f"{i + 2}-{end + 1}",
                total_sheets=total_sheets, sheet_row_count=n,
            ),
        ))
        if end >= n:
            break
        i += step

    return docs


def load(filename: str, data: bytes) -> List[Document]:
    try:
        sheets = _read_sheets(filename, data)
    except Exception as e:
        log.error(f"Excel load error ({filename}): {e}")
        return []

    # Drop sheets that are entirely empty.
    sheets = {name: df for name, df in sheets.items() if not df.dropna(how="all").empty}
    if not sheets:
        return []

    docs: List[Document] = []
    total_sheets = len(sheets)

    for idx, (sheet_name, raw_df) in enumerate(sheets.items()):
        df = raw_df.dropna(how="all").reset_index(drop=True)
        df.columns = _clean_columns(df.columns)

        truncated_from = None
        if len(df) > settings.EXCEL_MAX_ROWS_PER_SHEET:
            truncated_from = len(df)
            df = df.iloc[: settings.EXCEL_MAX_ROWS_PER_SHEET].reset_index(drop=True)

        docs.append(_build_summary_chunk(sheet_name, df, filename, idx, total_sheets, truncated_from))
        docs.extend(_build_row_group_chunks(sheet_name, df, filename, idx, total_sheets))

    log.info(f"Excel '{filename}': {total_sheets} sheet(s) -> {len(docs)} chunk(s)")
    return docs
