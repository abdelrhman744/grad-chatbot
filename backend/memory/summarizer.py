"""
summarizer.py

Updates the long-term conversation summary using an LLM. Decoupled from
any specific LLM client — it only requires an object exposing
`.generate(prompt: str) -> str` (see memory.llm_adapter.LLMTextGenerator).
"""

from __future__ import annotations


def _format_messages(messages: list[dict]) -> str:
    """Convert message dicts into readable 'Role: content' text."""
    formatted = []
    for message in messages:
        role = message["role"].capitalize()
        content = message["content"]
        formatted.append(f"{role}: {content}")
    return "\n\n".join(formatted)


def build_prompt(old_summary: str, messages: list[dict]) -> str:
    conversation = _format_messages(messages)

    if not old_summary:
        old_summary = "No previous summary."

    prompt = f"""
You are responsible for maintaining the long-term memory of a chatbot.

Current Summary
---------------
{old_summary}

Recent Conversation
-------------------
{conversation}

Task
----
Update the conversation summary using BOTH the current summary and the recent conversation.

Guidelines:
1. Preserve important long-term facts.
2. Preserve user goals.
3. Preserve important decisions.
4. Preserve unresolved tasks.
5. Remove duplicate information.
6. Ignore greetings and casual conversation.
7. Keep the summary under 300 words.

Return ONLY the updated summary.
"""

    return prompt.strip()


def update_summary(llm, old_summary: str, messages: list[dict]) -> str:
    """
    Generate an updated conversation summary.

    Args:
        llm: An object exposing `.generate(prompt: str) -> str`.
        old_summary: Previously stored summary.
        messages: Recent conversation messages.
    """
    prompt = build_prompt(old_summary, messages)
    updated_summary = llm.generate(prompt)
    return updated_summary.strip()
