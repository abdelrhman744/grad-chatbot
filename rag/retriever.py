"""
retriever.py

Retrieve relevant chunks using:

- Query Expansion (opt-in — see RAG_ENABLE_QUERY_EXPANSION)
- Dense Retrieval
- BM25
- Reciprocal Rank Fusion
- Re-ranking

Latency-oriented defaults
-------------------------
- RAG_FUSED_CANDIDATE_MULTIPLIER=1  (was 2)
- RAG_SEARCH_CANDIDATE_MULTIPLIER=1.5  (dense/sparse each fetch top_k * this)
- Query expansion off by default
- Cross-encoder skipped when fused set is already ≤ top_k
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

from .query_expansion import QueryExpander
from .hybrid_search import HybridSearch
from .reranker import Reranker
from .search import SearchEngine
from .bm25 import BM25Searcher
from .document_ranker import DocumentRanker

logger = logging.getLogger(__name__)

FUSED_CANDIDATE_MULTIPLIER = int(os.getenv("RAG_FUSED_CANDIDATE_MULTIPLIER", "1"))
SEARCH_CANDIDATE_MULTIPLIER = float(os.getenv("RAG_SEARCH_CANDIDATE_MULTIPLIER", "1.5"))

_search_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="retriever-search")

_LOG_LATENCY = os.getenv("RAG_LOG_LATENCY", "false").strip().lower() in ("1", "true", "yes")


class Retriever:

    def __init__(self, enable_expansion: bool | None = None):

        self.dense_search = SearchEngine()
        self.sparse_search = BM25Searcher()
        self.hybrid = HybridSearch()
        self.expander = QueryExpander()
        self.reranker = Reranker()
        self.document_ranker = DocumentRanker()

        if enable_expansion is not None:
            self.expansion_enabled = enable_expansion
        else:
            self.expansion_enabled = os.getenv(
                "RAG_ENABLE_QUERY_EXPANSION", "true"
            ).strip().lower() in ("1", "true", "yes")

    def should_expand(self, question: str) -> bool:
        if not self.expansion_enabled:
            return False

        question = question.strip().lower()

        greetings = {
            "hi", "hello", "hey", "thanks", "thank you",
            "bye", "goodbye", "who are you", "help",
        }
        if question in greetings:
            return False
        if len(question.split()) <= 2:
            return False
        return True

    def retrieve(self, question: str, top_k: int = 5):
        t_start = time.perf_counter()

        # -------------------------------------------------
        # Query Expansion (opt-in, usually off)
        # -------------------------------------------------
        if self.should_expand(question):
            t0 = time.perf_counter()
            search_query = self.expander.expand(question)
            if _LOG_LATENCY:
                logger.info(
                    "[latency] query_expansion=%.3fs",
                    time.perf_counter() - t0,
                )
        else:
            search_query = question

        # -------------------------------------------------
        # Dense + Sparse (concurrent)
        # -------------------------------------------------
        search_k = max(1, int(round(top_k * SEARCH_CANDIDATE_MULTIPLIER)))

        dense_future = _search_executor.submit(
            self.dense_search.search,
            query=search_query,
            top_k=search_k,
        )
        sparse_future = _search_executor.submit(
            self.sparse_search.search,
            query=search_query,
            top_k=search_k,
        )

        t0 = time.perf_counter()
        dense_results = dense_future.result()
        sparse_results = sparse_future.result()
        if _LOG_LATENCY:
            logger.info(
                "[latency] dense+sparse_parallel=%.3fs dense=%s sparse=%s",
                time.perf_counter() - t0,
                len(dense_results) if dense_results is not None else 0,
                len(sparse_results) if sparse_results is not None else 0,
            )

        # -------------------------------------------------
        # Reciprocal Rank Fusion
        # -------------------------------------------------
        t0 = time.perf_counter()
        fused_results = self.hybrid.fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=max(1, int(top_k * FUSED_CANDIDATE_MULTIPLIER)),
        )
        if _LOG_LATENCY:
            logger.info(
                "[latency] rrf_fuse=%.3fs fused=%s",
                time.perf_counter() - t0,
                len(fused_results),
            )

        # -------------------------------------------------
        # Re-rank (skip if already small enough)
        # -------------------------------------------------
        if len(fused_results) <= top_k:
            final_results = fused_results
            for document in final_results:
                document.setdefault("rerank_score", document.get("rrf_score", 0.0))
        else:
            final_results = self.reranker.rank(
                question=question,
                documents=fused_results,
                top_k=top_k,
            )

        self.document_ranker.rank(final_results)

        if _LOG_LATENCY:
            logger.info(
                "[latency] retrieve_total=%.3fs final=%s",
                time.perf_counter() - t_start,
                len(final_results),
            )

        return final_results
