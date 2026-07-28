"""
respond_tool.py

Terminal tool: answers directly from conversation memory without
querying the vector database. Used for greetings, small talk, and
follow-up questions the agent judges answerable from memory alone.
"""

from __future__ import annotations

import re
from typing import Iterator

from rag import prompt as prompt_lib
from rag.agent.schemas import ExecutionContext
from rag.generator import Generator


def _normalize_for_comparison(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


# Fallback shown when the model's answer turns out to be a verbatim (or
# near-verbatim) echo of the last stored assistant turn instead of a
# fresh answer to the current question — see _is_echo_of_last_answer.
_ECHO_FALLBACK = {
    "en": "I don't have new information in memory that answers this specific question.",
    "ar": "لا توجد لدي معلومة جديدة في الذاكرة تجيب على هذا السؤال تحديدًا.",
}


class RespondTool:

    def __init__(
        self,
        generator: Generator | None = None,
        memory_text_provider=None,
        last_assistant_provider=None,
    ):
        self.generator = generator or Generator()
        self._memory_text_provider = memory_text_provider or (lambda: "")
        # Optional: lets run()/stream_run() detect when the model just
        # copy-pasted the previous turn's answer instead of addressing the
        # current question (see _is_echo_of_last_answer below).
        self._last_assistant_provider = last_assistant_provider or (lambda: "")

    # =====================================================
    # Public Method
    # =====================================================

    def run(self, context: ExecutionContext, question: str) -> ExecutionContext:

        memory_text = self._memory_text_provider()
        language = context.language

        prompt = prompt_lib.build_memory_prompt(question, memory_text, language)

        raw_answer = self.generator.generate(prompt)
        answer = prompt_lib.clean_answer(raw_answer, language)

        context.answer = self._guard_against_echo(answer, language)

        context.observations.append({"tool": "respond", "status": "answered_from_memory"})

        return context

    # =====================================================
    # Echo guard
    # =====================================================

    def _guard_against_echo(self, answer: str, language: str) -> str:
        """
        Safety net for the exact failure mode reported: a weak model, given
        conversation memory that already contains a prior assistant reply,
        sometimes just copy-pastes that reply back as if it were a fresh
        answer to a DIFFERENT current question — instead of extracting or
        synthesizing a new one (build_memory_prompt now explicitly forbids
        this, but a prompt instruction alone isn't a guarantee with a small
        model). If that happens here, it's caught structurally instead of
        silently shipping the stale duplicate (and, worse, letting it get
        stored back into memory and compound further on the next turn).

        Deliberately conservative: only triggers on an exact or near-exact
        match (normalized whitespace/case) against the last stored
        assistant turn, not on topical similarity, so genuine short answers
        that happen to reuse a few words are never falsely flagged.
        """
        last_answer = self._last_assistant_provider()

        if not last_answer or not answer:
            return answer

        normalized_answer = _normalize_for_comparison(answer)
        normalized_last = _normalize_for_comparison(last_answer)

        if not normalized_answer:
            return answer

        is_exact_match = normalized_answer == normalized_last
        # Also catch the case where the new answer is the old one with a
        # little extra text tacked on (e.g. one more accumulated source
        # fragment) — containment in either direction, not just equality.
        is_containment = (
            len(normalized_answer) > 20
            and (normalized_answer in normalized_last or normalized_last in normalized_answer)
        )

        if is_exact_match or is_containment:
            return _ECHO_FALLBACK.get(language, _ECHO_FALLBACK["en"])

        return answer

    # =====================================================
    # Streaming variant
    # =====================================================

    def stream_run(self, context: ExecutionContext, question: str) -> Iterator[str]:
        memory_text = self._memory_text_provider()
        language = context.language

        prompt = prompt_lib.build_memory_prompt(question, memory_text, language)

        context.observations.append({"tool": "respond", "status": "answered_from_memory"})

        collected = []
        for delta in self.generator.stream(prompt):
            collected.append(delta)
            yield delta

        answer = prompt_lib.clean_answer("".join(collected), language)
        guarded = self._guard_against_echo(answer, language)

        if guarded != answer:
            # The stream already showed the raw (echoed) text to the user
            # by this point — that can't be un-sent. What we CAN still do
            # is (a) make sure the corrected, non-duplicate text is what
            # gets written to conversation memory (so the echo doesn't
            # compound into an even longer echo next turn), and (b) warn
            # the user inline that the reply above wasn't specific to
            # their question.
            note = (
                "\n\n⚠️ الرد أعلاه كان تكرارًا لإجابة سابقة ولا يخص سؤالك الحالي تحديدًا."
                if language == "ar"
                else "\n\n⚠️ The reply above repeated an earlier answer and doesn't specifically address your current question."
            )
            yield note
            context.answer = guarded
        else:
            context.answer = answer