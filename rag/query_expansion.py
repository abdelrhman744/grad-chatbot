"""
query_expansion.py

Expand the user's query using the LLM.
"""

from .llm import LLM


class QueryExpander:

    def __init__(self):
        self.llm = LLM()

    # =====================================================
    # Public Method
    # =====================================================

    def expand(self, question: str) -> str:

        prompt = f"""
You are an expert query expansion assistant for a multilingual
(Arabic and English) Retrieval-Augmented Generation (RAG) system.

Your task is to improve retrieval by expanding the user's search query.

User Question:
{question}

Instructions:

1. Keep the ORIGINAL question exactly as written.
2. Do NOT rewrite or remove any part of the original question.
3. Append important synonyms, related terminology,
   abbreviations, equivalent phrases and alternative wording.
4. Preserve names, IDs, course codes and technical terms.
5. Do NOT answer the question.
6. Return ONE expanded search query.
7. Do NOT use labels like "Original Question" or
   "Expanded Query".

Output:
"""

        expanded_query = self.llm.generate(prompt)

        return expanded_query.strip()