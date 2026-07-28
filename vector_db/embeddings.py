"""
embeddings.py

Generate embeddings for text chunks and structured documents.
"""

import uuid

from .embedding_service import EmbeddingService


class EmbeddingGenerator:

    def __init__(self):
        pass

    # =====================================================
    # OCR Chunks
    # =====================================================

    def embed(
        self,
        chunks: list[str]
    ) -> list[dict]:

        if not chunks:
            return []

        embeddings = EmbeddingService.embed_passages(
            chunks
        )

        documents = []

        for chunk, embedding in zip(chunks, embeddings):

            documents.append(

                {

                    "id": str(uuid.uuid4()),

                    "text": chunk,

                    "embedding": embedding.tolist()

                }

            )

        return documents

    # =====================================================
    # Structured Documents (Excel / Tables)
    # =====================================================

    def embed_documents(
        self,
        documents: list[dict]
    ) -> list[dict]:

        if not documents:
            return []
 
        texts = [

            document["text"]

            for document in documents

        ]

        embeddings = EmbeddingService.embed_passages(
            texts
        )

        output = []

        for document, embedding in zip(
            documents,
            embeddings
        ):

            output.append(

                {

                    "id": str(uuid.uuid4()),

                    "text": document["text"],

                    "embedding": embedding.tolist(),

                    "metadata": document["metadata"]

                }

            )

        return output