"""
hybrid_search.py

Fuse dense and sparse retrieval results using Reciprocal Rank Fusion (RRF).
"""

from .result_formatter import ResultFormatter


class HybridSearch:

    def __init__(self):

        self.rrf_k = 60

    # =====================================================
    # Public Method
    # =====================================================

    def fuse(
        self,
        dense_results,
        sparse_results,
        top_k: int = 10
    ):

        scores = {}

        objects = {}

        # -------------------------------------------------
        # Dense Results
        # -------------------------------------------------

        for rank, point in enumerate(dense_results, start=1):

            chunk_id = str(point.id)

            scores[chunk_id] = scores.get(chunk_id, 0) + (
                1 / (self.rrf_k + rank)
            )

            objects[chunk_id] = ResultFormatter.from_qdrant(point)

        # -------------------------------------------------
        # Sparse Results
        # -------------------------------------------------

        for rank, document in enumerate(sparse_results, start=1):

            chunk_id = document["id"]

            scores[chunk_id] = scores.get(chunk_id, 0) + (
                1 / (self.rrf_k + rank)
            )

            if chunk_id not in objects:

                objects[chunk_id] = ResultFormatter.from_bm25(
                    document
                )

        # -------------------------------------------------
        # Final Ranking
        # -------------------------------------------------

        ranked = sorted(

            scores.items(),

            key=lambda item: item[1],

            reverse=True

        )

        results = []

        for chunk_id, score in ranked[:top_k]:

            document = objects[chunk_id]

            document["rrf_score"] = score

            results.append(document)

        return results