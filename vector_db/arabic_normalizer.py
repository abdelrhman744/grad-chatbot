"""
arabic_normalizer.py
"""

import re


class ArabicNormalizer:

    @staticmethod
    def normalize(text: str) -> str:

        if not text:
            return ""

        text = re.sub(r"[ًٌٍَُِّْـ]", "", text)
        text = re.sub(r"[أإآ]", "ا", text)

        text = text.replace("ى", "ي")
        text = text.replace("ؤ", "و")
        text = text.replace("ئ", "ي")
        text = text.replace("ة", "ه")

        text = re.sub(r"\s+", " ", text)

        return text.strip()