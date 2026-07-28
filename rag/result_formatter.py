"""
result_formatter.py

Convert search results into a unified format.
"""


class ResultFormatter:

    @staticmethod
    def from_qdrant(point):

        payload = point.payload

        return {

            "id": str(point.id),

            "text": payload.get("text", ""),

            "metadata": {

                "document_id": payload.get("document_id"),

                "title": payload.get("title"),

                "document_type": payload.get("document_type"),

                "uploaded_by": payload.get("uploaded_by"),

                "roles": payload.get("roles"),

                "document_scope": payload.get("document_scope"),

                # Location fields for page-level source citations. These are
                # passed through as-is if the indexer set them; when a
                # document has no page concept (e.g. a spreadsheet), both are
                # simply absent and citations fall back to the title only.
                "page": payload.get("page") or payload.get("page_number"),

                "chunk_index": payload.get("chunk_index")

            },

            "score": point.score

        }

    @staticmethod
    def from_bm25(document):

        metadata = document.get("metadata", {})

        return {

            "id": document["id"],

            "text": document["text"],

            "metadata": {

                "document_id": metadata.get("document_id"),

                "title": metadata.get("title"),

                "document_type": metadata.get("document_type"),

                "uploaded_by": metadata.get("uploaded_by"),

                "roles": metadata.get("roles"),

                "document_scope": metadata.get("document_scope"),

                "page": metadata.get("page") or metadata.get("page_number"),

                "chunk_index": metadata.get("chunk_index")

            },

            "score": document["score"]

        }