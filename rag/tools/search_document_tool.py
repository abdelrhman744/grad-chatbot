"""
search_document_tool.py

Tool for searching documents.
"""

from ..retriever import Retriever


class SearchDocumentTool:

    def __init__(self):

        self.retriever = Retriever()

        self.name = "search_documents"

        self.description = (
            "Search for documents related to the user's request."
        )

    # =====================================================
    # Public Method
    # =====================================================

    def run(
        self,
        query: str,
        top_k: int = 10
    ):

        documents = self.retriever.retrieve(
            question=query,
            top_k=top_k
        )

        results = []

        seen_titles = set()

        for document in documents:

            metadata = document.get("metadata", {})

            title = metadata.get("title", "Unknown")

            if title in seen_titles:
                continue

            seen_titles.add(title)

            results.append({

                "title": title,

                "document_id": metadata.get("document_id"),

                "document_type": metadata.get("document_type"),

                "uploaded_by": metadata.get("uploaded_by"),

                "roles": metadata.get("roles"),

                "document_scope": metadata.get("document_scope")

            })

        return results