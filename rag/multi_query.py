"""
multi_query.py

Generate multiple retrieval queries.
"""

from .llm import LLM


class MultiQueryGenerator:

    def __init__(self):

        self.llm = LLM()

    # =====================================================
    # Public Method
    # =====================================================

    def generate(self, expanded_query: str):

        prompt = f"""
You are an expert retrieval planner for a multilingual (Arabic and English) RAG system.

Generate four diverse search queries from the expanded query below.

Expanded Query:
"{expanded_query}"

Instructions:
1. Keep the original intent.
2. Each query should retrieve different but relevant documents.
3. Use different wording and perspectives.
4. Do NOT answer the question.
5. Return exactly four queries.
6. One query per line.
7. No numbering.
8. No bullets.
9. No explanations.

Search Queries:
"""

        response = self.llm.generate(prompt)

        queries = []

        for line in response.split("\n"):

            line = line.strip()

            if line:
                queries.append(line)

        return queries