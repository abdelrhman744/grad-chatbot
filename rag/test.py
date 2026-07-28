from .bm25 import BM25Searcher

bm25 = BM25Searcher()

results = bm25.search(
    "الاحتباس ",
    top_k=5
)

for result in results:

    print(result["score"])
    print(result["title"])
    print(result["text"][:150])
    print("-" * 50)