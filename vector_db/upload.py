"""
upload.py

Upload embeddings to Qdrant.
"""

from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)


class QdrantUploader:

    def __init__(self):

        # Connect to local Qdrant
        self.client = QdrantClient(
        path=str(Path(__file__).parent / "qdrant_db")
                            )
        self.collection_name = "documents"

        self._create_collection()

    # =====================================================
    # Public Methods
    # =====================================================

    def upload(self, documents: list[dict]):

        points = []

        for index, document in enumerate(documents):

            metadata = document["metadata"]

            
            point = PointStruct(
                id=document["id"],
                vector=document["embedding"],
                payload={
                    "text": document["text"],
                    **metadata
                }
            )

            points.append(point)

        self.client.upsert(

            collection_name=self.collection_name,

            points=points

        )

        print(f"Uploaded {len(points)} chunks.")

    # =====================================================
    # Private Methods
    # =====================================================

    def _create_collection(self):

        collections = self.client.get_collections().collections

        names = [collection.name for collection in collections]

        if self.collection_name not in names:

            self.client.create_collection(

                collection_name=self.collection_name,

                vectors_config=VectorParams(

                    size=1024,
                    distance=Distance.COSINE

                )

            )
    def get_first_points(self, limit=4):

        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        return points            