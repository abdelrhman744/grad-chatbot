"""
ensemble.py

Combines the OCR results from multiple preprocessing methods / PSM
modes into the final text for a page.

Two strategies are supported (Config.ENSEMBLE_STRATEGY):

- "best": pick the single highest-scoring attempt using a weighted
  confidence + length + Arabic-script + digit score. Prefer this for
  clean lecture PDFs — avoids merging contradictory misreads.
- "merge": take every OCR attempt's text and merge them line-by-line,
  deduplicating near-identical lines and keeping the longer version
  of each. Higher recall, but can mix good and bad readings of numbers.

After selection, a light cleaner removes Latin junk and a number-recovery
pass patches short/truncated measurements using better readings from
other OCR attempts (e.g. 6 → 146.6 when another attempt saw 146.6).
"""

from __future__ import annotations

import re

from .config import Config
from .models import PageData

# Arabic letter range — used to prefer results that actually contain Arabic
_AR_RE = re.compile(r"[\u0600-\u06FF]")

# Likely OCR garbage tokens that appear in bad Arabic / mixed scans
_GARBAGE_RE = re.compile(
    r"\b(?:ALS|ple|lib|Lb|Vy20|Ijx0|Iza|Cod|hel|bell|Mol|Vol|gil|eject|sits|"
    r"x0|xz0|khx0|>99|<<|>>)\b",
    re.IGNORECASE,
)

# Isolated short Latin tokens left over from mixed-script OCR
_LATIN_JUNK_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{1,4}(?![A-Za-z0-9])")

# Common OCR artifacts around numbers in Arabic tables
_NUM_ARTIFACT_RE = re.compile(
    r"(?:"
    r"[>＜＜«»‹›]\s*\d+"          # >99, «12 etc.
    r"|\d+\s*[xX×]\s*0"          # 146.6 x0
    r"|[xX×]\s*0"                # lone x0
    r"|\b0\s*[xX×]"              # 0x
    r")"
)

# Any number token (integer or decimal)
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


class Ensemble:
    """
    Chooses or builds the final OCR text for a page from all of its
    OCRResults, then cleans and recovers truncated numbers.
    """

    def process(self, page: PageData) -> PageData:
        if not page.ocr_results:
            return page

        strategy = (Config.ENSEMBLE_STRATEGY or "best").strip().lower()

        if strategy == "merge":
            self._process_merge(page)
        else:
            self._process_best(page)

        if page.text:
            page.text = self._clean_text(page.text)
            page.text = self._recover_numbers(page)

        return page

    # ======================================================
    # Best-of strategy (recommended)
    # ======================================================

    def _process_best(self, page: PageData) -> None:
        best_result = None
        best_score = float("-inf")

        for result in page.ocr_results.values():
            if not (result.text or "").strip():
                continue
            score = self._calculate_score(result)
            if score > best_score:
                best_score = score
                best_result = result

        if best_result is not None:
            page.text = best_result.text
            page.confidence = best_result.confidence

    def _calculate_score(self, result) -> float:
        """
        Weighted score for one OCR attempt.

        - Confidence from Tesseract
        - Length (prefer more complete pages)
        - Arabic character ratio (prefer real Arabic script)
        - Digit / decimal / multi-digit density (protect tables)
        - Heavy penalty for known garbage + leftover Latin junk
        """
        text = result.text or ""
        if not text.strip():
            return float("-inf")

        confidence_score = result.confidence * Config.ENSEMBLE_CONFIDENCE_WEIGHT
        length_score = len(text) * Config.ENSEMBLE_LENGTH_WEIGHT

        ar_chars = len(_AR_RE.findall(text))
        total_letters = max(len(re.findall(r"[^\W\d_]", text, flags=re.UNICODE)), 1)
        arabic_ratio = ar_chars / total_letters
        arabic_bonus = arabic_ratio * 15.0

        digit_chars = len(re.findall(r"\d", text))
        decimal_nums = len(re.findall(r"\d+\.\d+", text))
        multi_digit = len(re.findall(r"\d{2,}", text))
        digit_bonus = digit_chars * 0.8 + decimal_nums * 15.0 + multi_digit * 4.0

        garbage_hits = len(_GARBAGE_RE.findall(text))
        latin_junk = len(_LATIN_JUNK_RE.findall(text))
        num_artifacts = len(_NUM_ARTIFACT_RE.findall(text))
        garbage_penalty = (
            garbage_hits * 8.0
            + latin_junk * 3.0
            + num_artifacts * 10.0
        )

        return (
            confidence_score
            + length_score
            + arabic_bonus
            + digit_bonus
            - garbage_penalty
        )

    # ======================================================
    # Merge strategy
    # ======================================================

    def _process_merge(self, page: PageData) -> None:
        results = [
            result for result in page.ocr_results.values()
            if (result.text or "").strip()
        ]
        if not results:
            return

        ranked = sorted(results, key=self._calculate_score, reverse=True)
        merged_text = self._merge_texts([r.text for r in ranked])

        page.text = merged_text
        page.confidence = sum(r.confidence for r in results) / len(results)

    def _merge_texts(self, texts) -> str:
        seen = set()
        lines = []

        for text in texts:
            for line in text.splitlines():
                key = re.sub(r"\s+", " ", line.strip().lower())
                key = re.sub(r"\b[a-z]{2,4}\b", "", key)
                key = re.sub(r"\s+", " ", key).strip()
                if key and key not in seen:
                    seen.add(key)
                    lines.append(line.strip())

        return "\n".join(lines)

    # ======================================================
    # Post-OCR cleaning
    # ======================================================

    def _clean_text(self, text: str) -> str:
        """
        Remove common mixed-script OCR artifacts while preserving
        Arabic text and real numbers (including decimals).
        """
        if not text:
            return text

        lines = []
        for line in text.splitlines():
            cleaned = line

            cleaned = _GARBAGE_RE.sub(" ", cleaned)
            cleaned = _NUM_ARTIFACT_RE.sub(" ", cleaned)
            cleaned = _LATIN_JUNK_RE.sub(" ", cleaned)
            cleaned = re.sub(r"[\u200e\u200f\u202a-\u202e]", "", cleaned)
            cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()

            if cleaned and not re.fullmatch(r"[\W_]+", cleaned):
                lines.append(cleaned)

        return "\n".join(lines)

    # ======================================================
    # Number recovery across OCR attempts
    # ======================================================

    def _recover_numbers(self, page: PageData) -> str:
        """
        Collect the best (longest / most precise) number readings from
        every OCR attempt, then replace truncated numbers in the final
        text when a clearly better candidate exists.

        Example failure mode this fixes:
            final text has "6 مترًا" while another attempt had "146.6"
        """
        text = page.text or ""
        if not text.strip() or not page.ocr_results:
            return text

        # Gather all number tokens from every attempt, preferring longer ones
        candidates: dict[str, str] = {}

        for result in page.ocr_results.values():
            for m in _NUMBER_RE.finditer(result.text or ""):
                token = m.group(0)
                # Skip pure zeros / single-digit noise unless decimal
                if len(token) == 1 and "." not in token:
                    continue
                key = token
                int_part = token.split(".")[0]
                if int_part not in candidates or len(token) > len(candidates[int_part]):
                    candidates[int_part] = token
                if key not in candidates or len(token) > len(candidates[key]):
                    candidates[key] = token

        if not candidates:
            return text

        def replace_match(m: re.Match) -> str:
            token = m.group(0)
            int_part = token.split(".")[0]

            # Already a good multi-digit or decimal number — keep it
            if len(token) >= 3 or ("." in token and len(token) >= 3):
                return token

            # Short token (e.g. "6") — look for a longer candidate that
            # ends with the same digit(s) (146.6 ends with 6, 65 ends with 5)
            best = None
            for cand in candidates.values():
                if cand == token:
                    continue
                if cand.endswith(token) and len(cand) > len(token):
                    if best is None or len(cand) > len(best):
                        best = cand
            return best if best is not None else token

        return _NUMBER_RE.sub(replace_match, text)