"""
question_splitter.py

Splits a user message that contains more than one question into
separate, self-contained sub-questions for retrieval.

An earlier version split purely on '؟'/'?' characters. That breaks in
two common cases:

  1. The user asks multiple questions with NO question mark anywhere
     ("من مؤسس الشركة ومتى تأسست", or two sentences in a row with
     no punctuation at all). Splitting on '؟'/'?' misses these
     entirely and the whole message is treated as one question.

  2. A trailing fragment like "ومتى؟" ("and when?") has no subject of
     its own once cut off at a '؟'. Retrieving it standalone (just
     "and when?") returns noise, since the retriever has nothing to
     search for.

This version instead detects sub-question boundaries using a list of
Arabic/English question-words, not punctuation. Punctuation is still
respected when present (a '؟'/'?' ends the fragment that started at
the previous boundary), but is no longer required to detect that the
message is compound.

Two tiers of question-words are used:

  - STRONG words (متى/أين/كيف/لماذا/هل/كم, when/where/why/how/who...)
    are unambiguous enough that they mark a new sub-question wherever
    they appear as a standalone word.
  - WEAK words (من/ما/ماذا, what/which/whose) are common outside
    questions too ("من" = "who" but also the preposition "from"; "ما"
    doubles as a negation/relative particle in Arabic). These only
    count as a boundary at the very start of the message or right
    after a connector ("و", "and", "ثم", a comma, ...), which is a
    much more reliable signal that a new clause is starting.

Once boundaries are found, any fragment too short to carry its own
topic (e.g. "ومتى؟") has the previous fragment's topic appended, so
every sub-question returned is self-contained enough to retrieve on
its own.
"""

from __future__ import annotations

import re

_MIN_FRAGMENT_LENGTH = 4  # skip tiny/empty leftover fragments entirely
_MIN_TOPIC_WORDS = 3  # fragments with fewer content words than this get context prepended

# Short messages ("مرحبا كيف حالك", "hi, how are you") are almost always
# a greeting, not a genuine compound question — "كيف"/"how" shows up in
# plenty of small talk. Require a minimum overall word count before
# even attempting to split, so greetings are left as a single message
# and handled by the planner's normal "respond" path instead of being
# pulled apart and sent to retrieval.
_MIN_TOTAL_WORDS = 6

_LEADING_JUNK_RE = re.compile(r"^[\s,،]+")

# Question words unambiguous enough to mark a new sub-question wherever
# they appear as a standalone word (no connector required).
_STRONG_QUESTION_WORDS = [
    "متى", "أين", "وين", "كيف", "لماذا", "ليش", "هل", "كم",
    "who", "when", "where", "why", "how",
]

# Question words that are also common non-interrogative words in
# ordinary sentences ("من" = "from", "ما" = negation/relative particle,
# "what"/"which" show up in embedded clauses). Only treated as a
# boundary at the start of the message or right after a connector.
_WEAK_QUESTION_WORDS = [
    "من", "ما", "ماذا", "أي",
    "which", "what", "whose",
]

# Words/punctuation that, immediately before a weak question word,
# signal "a new clause is starting" rather than the word being
# embedded inside the current one.
_CONNECTORS = ["و", "ثم", "أيضا", "كذلك", "and", "also", "then"]

_ALL_QUESTION_WORDS = _STRONG_QUESTION_WORDS + _WEAK_QUESTION_WORDS

_FLAGS = re.IGNORECASE | re.UNICODE


def _alternation(words: list[str]) -> str:
    # Longest-first so e.g. "ماذا" matches before "ما" would greedily
    # eat only part of it.
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


# Plain standalone occurrence, e.g. "when was it founded" or
# "اخبرني أين تقع" — the word is separated from what comes before it
# by whitespace/punctuation, so a normal \b boundary is enough.
_STRONG_RE = re.compile(rf"\b(?:{_alternation(_STRONG_QUESTION_WORDS)})\b", _FLAGS)

# Arabic attaches connectors like "و" (and) directly to the following
# word with NO space ("ومتى" = "و" + "متى"), so from regex's point of
# view "ومتى" is a single token and \b never fires between "و" and
# "متى". Catch that case explicitly for strong words too, the same
# way it's already required for weak words below.
_STRONG_ATTACHED_RE = re.compile(
    rf"(?:^|(?:{_alternation(_CONNECTORS)})\s*|[,،]\s*)"
    rf"(?:{_alternation(_STRONG_QUESTION_WORDS)})\b",
    _FLAGS,
)

# A handful of common Arabic idioms where "من"/"ما" aren't actually
# interrogative ("من فضلك" = "please", "ما شاء الله" = an exclamation),
# excluded so they don't get mistaken for the start of a question.
_NON_INTERROGATIVE_FOLLOWUPS = ["فضلك", "شاء الله", "شأنك", "شأنه"]

_WEAK_RE = re.compile(
    rf"(?:^|(?:{_alternation(_CONNECTORS)})\s*|[,،]\s*)"
    rf"(?:{_alternation(_WEAK_QUESTION_WORDS)})\b"
    rf"(?!\s*(?:{_alternation(_NON_INTERROGATIVE_FOLLOWUPS)}))",
    _FLAGS,
)

# Used when trimming a question word (and any leading connector) off
# the front of a fragment to find its "topic" — see _extract_topic.
_LEADING_QWORD_RE = re.compile(
    rf"^(?:(?:{_alternation(_CONNECTORS)})\s*)?(?:{_alternation(_ALL_QUESTION_WORDS)})\b\s*",
    _FLAGS,
)

_TRAILING_PUNCT_RE = re.compile(r"[؟?.!\s]+$")


def split_questions(text: str) -> list[str]:
    """
    Split `text` into self-contained sub-questions.

    Returns a list with the original text unchanged (as a single-item
    list) if fewer than 2 real sub-questions are found, so callers can
    treat "not actually compound" text exactly as before.
    """
    text = (text or "").strip()

    if not text:
        return [text]

    if len(text.split()) < _MIN_TOTAL_WORDS:
        return [text]

    boundaries = _find_boundaries(text)

    if len(boundaries) < 2:
        return [text]

    raw_fragments = []
    for index, start in enumerate(boundaries):
        end = boundaries[index + 1] if index + 1 < len(boundaries) else len(text)
        fragment = _LEADING_JUNK_RE.sub("", text[start:end]).strip()
        if fragment:
            raw_fragments.append(fragment)

    fragments = [fragment for fragment in raw_fragments if len(fragment) >= _MIN_FRAGMENT_LENGTH]

    if len(fragments) < 2:
        return [text]

    return _contextualize(fragments)


def _find_boundaries(text: str) -> list[int]:
    """
    Collect every position in `text` where a new sub-question clause
    starts (position 0 is always included so leading text before the
    first detected question-word isn't dropped), then drop boundaries
    that are too close together to represent a real second clause.
    """
    positions = {0}

    for match in _STRONG_RE.finditer(text):
        positions.add(match.start())

    for match in _STRONG_ATTACHED_RE.finditer(text):
        positions.add(match.start())

    for match in _WEAK_RE.finditer(text):
        positions.add(match.start())

    sorted_positions = sorted(positions)

    filtered = [sorted_positions[0]]
    for pos in sorted_positions[1:]:
        if pos - filtered[-1] >= _MIN_FRAGMENT_LENGTH:
            filtered.append(pos)

    return filtered


def _extract_topic(fragment: str) -> str:
    """
    Best-effort extraction of the "topic" of a fragment — everything
    after its leading connector/question word — used to fill in a
    short trailing fragment that has no topic of its own (e.g. "ومتى؟").
    """
    stripped = _TRAILING_PUNCT_RE.sub("", fragment).strip()

    stripped = _LEADING_QWORD_RE.sub("", stripped, count=1).strip()

    # Drop a common Arabic linking word ("هو"/"هي") if it's now the
    # first word left, so what remains is the actual noun phrase.
    stripped = re.sub(r"^(?:هو|هي)\s+", "", stripped).strip()

    return stripped


def _contextualize(fragments: list[str]) -> list[str]:
    """
    Fragments with too few content words to be retrieved on their own
    (e.g. "ومتى؟" / "and when?") get the previous fragment's topic
    appended, so retrieval has something concrete to search for.
    """
    result: list[str] = []
    previous_topic = ""

    for fragment in fragments:
        topic = _extract_topic(fragment)
        content_word_count = len(topic.split())

        if content_word_count < _MIN_TOPIC_WORDS and previous_topic:
            trimmed = _TRAILING_PUNCT_RE.sub("", fragment).strip()
            fragment = f"{trimmed} {previous_topic}؟"

        result.append(fragment)

        # A weak fragment shouldn't overwrite a good topic for the
        # *next* fragment — only update when this one actually had
        # enough content of its own.
        if content_word_count >= _MIN_TOPIC_WORDS:
            previous_topic = topic

    return result
