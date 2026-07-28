"""
document_ranker.py

Ranks documents instead of individual chunks.
"""

from collections import defaultdict


class DocumentRanker:

    def rank(
        self,
        chunks: list[dict]
    ) -> list[dict]:

        if not chunks:
            return []

        # --------------------------------------------
        # Group chunks by document_id
        # --------------------------------------------

        grouped = defaultdict(list)

        for chunk in chunks:

            document_id = chunk["metadata"]["document_id"]

            grouped[document_id].append(chunk)

        # --------------------------------------------
        # Score each document
        # --------------------------------------------

        ranked_documents = []

        for document_id, document_chunks in grouped.items():

            # Use the highest reranker score
            score = max(

                chunk["rerank_score"]

                for chunk in document_chunks

            )

            ranked_documents.append(

                {

                    "document_id": document_id,

                    "score": score,

                    "metadata": document_chunks[0]["metadata"],

                    "chunks": document_chunks

                }

            )

        # --------------------------------------------
        # Sort documents
        # --------------------------------------------

        ranked_documents.sort(

            key=lambda x: x["score"],

            reverse=True

        )

        return ranked_documents