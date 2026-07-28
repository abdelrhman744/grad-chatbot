"""
structured_chunker.py

Converts structured Excel sheets into semantic text chunks
ready for embedding.
"""


class StructuredChunker:

    def __init__(
        self,
        rows_per_chunk: int = 30
    ):

        self.rows_per_chunk = rows_per_chunk

    # =====================================================
    # Public Method
    # =====================================================

    def chunk(
        self,
        workbook: dict
    ) -> list[dict]:

        chunks = []

        metadata = workbook["metadata"]

        filename = workbook["filename"]

        for sheet in workbook["sheets"]:

            sheet_chunks = self._chunk_sheet(
                filename=filename,
                metadata=metadata,
                sheet=sheet
            )

            chunks.extend(sheet_chunks)

        return chunks

    # =====================================================
    # Private Methods
    # =====================================================

    def _chunk_sheet(
        self,
        filename: str,
        metadata: dict,
        sheet: dict
    ) -> list[dict]:

        rows = sheet["rows"]

        columns = sheet["columns"]

        sheet_name = sheet["sheet_name"]

        documents = []

        for start in range(
            0,
            len(rows),
            self.rows_per_chunk
        ):

            batch = rows[
                start:start + self.rows_per_chunk
            ]

            text = self._build_text(
                sheet_name,
                columns,
                batch
            )

            documents.append(

                {
                    "text": text,

                    "metadata": {

                        **metadata,

                        "filename": filename,

                        "sheet": sheet_name,

                        "content_type": "table"

                    }

                }

            )

        return documents

    # =====================================================

    def _build_text(
        self,
        sheet_name,
        columns,
        rows
    ):

        lines = []

        lines.append(
            f"Sheet: {sheet_name}"
        )

        lines.append("")

        lines.append(
            "Columns:"
        )

        lines.append(
            ", ".join(
                map(str, columns)
            )
        )

        lines.append("")

        lines.append("Rows:")

        lines.append("")

        for row in rows:

            for column, value in zip(
                columns,
                row
            ):

                lines.append(
                    f"{column}: {value}"
                )

            lines.append("----------------")

        return "\n".join(lines)