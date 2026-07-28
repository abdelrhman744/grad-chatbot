"""
bm25.py

Keyword search using Whoosh BM25.
"""

import threading
from pathlib import Path

from whoosh.index import open_dir
from whoosh.qparser import MultifieldParser
from whoosh.qparser import OrGroup

from vector_db.arabic_normalizer import ArabicNormalizer

# Cheaper to (re)open than the embedding/reranker models, but there's
# still no reason to reopen the index directory for every new
# conversation's Retriever() — share one Index object per process.
_index = None
_init_lock = threading.Lock()


def _get_index():
    global _index
    if _index is None:
        with _init_lock:
            if _index is None:
                base_dir = Path(__file__).resolve().parent.parent
                index_dir = base_dir / "vector_db" / "whoosh_index"
                _index = open_dir(index_dir)
    return _index


class BM25Searcher:

    def __init__(self):

        self.index = _get_index()

    # =====================================================
    # Helpers
    # =====================================================

    @staticmethod
    def _clean_query(query: str) -> str:

        query = ArabicNormalizer.normalize(query)

        # Escape Whoosh special characters
        special = r'+-&|!(){}[]^"~*?:\/'

        for ch in special:
            query = query.replace(ch, " ")

        query = " ".join(query.split())

        return query

    # =====================================================
    # Public
    # =====================================================

    def search(
        self,
        query: str,
        top_k: int = 10
    ):

        query = self._clean_query(query)

        parser = MultifieldParser(

            ["title", "text"],

            schema=self.index.schema,

            group=OrGroup.factory(0.9)

        )

        parsed_query = parser.parse(query)

        results = []

        with self.index.searcher() as searcher:

            hits = searcher.search(
                parsed_query,
                limit=top_k
            )

            for hit in hits:

                results.append({

                    "id": hit["id"],

                    "score": float(hit.score),

                    "text": hit["text"],

                    "metadata": {

                        "title": hit["title"],

                        "document_id": hit.get("document_id", ""),

                        "document_type": hit.get("document_type", ""),

                        "uploaded_by": hit.get("uploaded_by", ""),

                        "roles": hit.get("roles", ""),

                        "document_scope": hit.get("document_scope", "")

                    }

                })

        return results