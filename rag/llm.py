"""
llm.py

Groq LLM client.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


class LLM:

    def __init__(self):

        # Load .env from rag/ first, then project root
        load_dotenv(Path(__file__).parent / ".env")
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")

        # API Key
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY not found.")

        # Groq Client
        self.client = Groq(api_key=api_key)

        # Model
        self.model = "llama-3.3-70b-versatile"

    # =====================================================
    # Public Method
    # =====================================================

    def generate(self, prompt: str) -> str:

        response = self.client.chat.completions.create(

            model=self.model,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2
        )

        return response.choices[0].message.content