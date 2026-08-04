"""
fact_extractor.py

Updates the long-term memory fact store using an LLM. Replaces the old
single-paragraph summarizer (summarizer.py): instead of rewriting one prose
blob from scratch every time — which let earlier facts silently drop and
let duplicate information accumulate across many rewrites, since nothing
but LLM instruction-following enforced either constraint — this asks for a
small set of discrete, categorized facts plus an explicit removal list,
and leaves deduplication/merging/capping to code (see
memory.summary_memory.FactStore), not to the model's best effort.

Decoupled from any specific LLM client — only requires an object exposing
`.generate(prompt: str) -> str` (see memory.llm_adapter.LLMTextGenerator).
"""

from __future__ import annotations

import json
import re

_VALID_CATEGORIES = {"preference", "decision", "task", "fact"}


def _format_messages(messages: list[dict]) -> str:
    """Convert message dicts into readable 'Role: content' text."""
    formatted = []
    for message in messages:
        role = message["role"].capitalize()
        content = message["content"]
        formatted.append(f"{role}: {content}")
    return "\n\n".join(formatted)


def _format_facts(facts: list[dict]) -> str:
    if not facts:
        return "No facts recorded yet."
    return "\n".join(f"- [{f.get('category', 'fact')}] {f['text']}" for f in facts)


def build_prompt(existing_facts: list[dict], messages: list[dict]) -> str:
    conversation = _format_messages(messages)
    facts_text = _format_facts(existing_facts)

    prompt = f"""
You are responsible for maintaining the long-term memory of a chatbot, as a
small set of discrete facts rather than one long paragraph.

Existing Facts
---------------
{facts_text}

Recent Conversation
-------------------
{conversation}

Task
----
Return an updated fact list using BOTH the existing facts and the recent
conversation.

Guidelines:
1. Keep facts atomic — one fact per item (a user preference, a decision, an
   unresolved task, an important detail) — not a paragraph.
2. Preserve important long-term facts, user goals, and decisions from the
   existing list unless they are resolved or superseded.
3. Do not restate information that's already an existing fact in different
   words. If new conversation adds detail to an existing fact, return one
   updated/merged fact for it instead of two overlapping ones.
4. Ignore greetings and casual small talk — do not turn them into facts.
5. Put resolved tasks or information that's no longer relevant into "remove"
   (quoting the existing fact's text) instead of carrying it forward.
6. Each fact needs a "category" (one of: preference, decision, task, fact)
   and an "importance" score from 1 (trivial) to 5 (critical to remember).

Return ONLY a JSON object in this exact shape — no explanation, no Markdown:
{{"facts": [{{"text": "...", "category": "preference|decision|task|fact", "importance": 1}}],
  "remove": ["<verbatim text of an existing fact to drop>"]}}
"""
    return prompt.strip()


def _parse_json_object(raw: str) -> dict:
    """Best-effort JSON parsing: strips Markdown code fences and pulls out
    the first {...} block if the model added stray text around it."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _clean_facts(raw_facts) -> list[dict]:
    if not isinstance(raw_facts, list):
        return []

    cleaned = []
    for f in raw_facts:
        if not isinstance(f, dict):
            continue
        text = str(f.get("text", "")).strip()
        if not text:
            continue

        category = str(f.get("category", "fact")).strip().lower()
        if category not in _VALID_CATEGORIES:
            category = "fact"

        try:
            importance = int(f.get("importance", 3))
        except (TypeError, ValueError):
            importance = 3
        importance = max(1, min(5, importance))

        cleaned.append({"text": text, "category": category, "importance": importance})

    return cleaned


def extract_facts(llm, existing_facts: list[dict], messages: list[dict]) -> dict:
    """
    Ask the LLM for an updated fact list.

    Returns {"facts": [{"text", "category", "importance"}, ...], "remove": [str, ...]}.
    The caller (memory.summary_memory.FactStore) is responsible for merging,
    deduplicating, and capping — this function only validates/normalizes the
    LLM's structured output.
    """
    prompt = build_prompt(existing_facts, messages)

    try:
        raw = llm.generate(prompt)
    except Exception:
        return {"facts": [], "remove": []}

    data = _parse_json_object(raw)

    facts = _clean_facts(data.get("facts"))
    remove = data.get("remove") if isinstance(data.get("remove"), list) else []
    remove = [str(r).strip() for r in remove if str(r).strip()]

    return {"facts": facts, "remove": remove}
