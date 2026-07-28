"""
summarizer.py

Builds the prompt used to update a conversation's long-term summary and
asks the LLM adapter to produce the new summary text.
"""

from __future__ import annotations


SUMMARY_PROMPT_TEMPLATE = """You maintain a running summary of a conversation between a user and an AI assistant.

Update the OLD SUMMARY so it also reflects the NEW MESSAGES below.

Rules:
- Keep it factual and concise (max 6-8 sentences).
- Preserve important facts, names, numbers, and decisions from the old summary.
- Do not invent information that isn't in the old summary or new messages.
- Write the summary in English regardless of the conversation's language.
- Return ONLY the updated summary text, nothing else.

OLD SUMMARY:
{old_summary}

NEW MESSAGES:
{new_messages}

UPDATED SUMMARY:"""


def build_summary_prompt(old_summary: str, new_messages: list[dict]) -> str:
    formatted_messages = "\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in new_messages
    )

    return SUMMARY_PROMPT_TEMPLATE.format(
        old_summary=old_summary.strip() or "(none yet)",
        new_messages=formatted_messages.strip() or "(none)",
    )


class Summarizer:

    def __init__(self, llm_adapter):
        self.llm_adapter = llm_adapter

    def summarize(self, old_summary: str, new_messages: list[dict]) -> str:
        if not new_messages:
            return old_summary

        prompt = build_summary_prompt(old_summary, new_messages)

        try:
            updated_summary = self.llm_adapter.generate(prompt)
            return updated_summary.strip() or old_summary
        except Exception:
            # If summarization fails, keep the old summary rather than
            # losing memory entirely.
            return old_summary
