"""
prompt.py

Builds the prompts sent to the LLM for answer generation, and cleans up
the raw model output before it's shown to the user. Shared by
generate_tool, respond_tool, summarize_tool, and compare_tool so prompt
wording and cleanup only need to live in one place.
"""

from __future__ import annotations

import re
from collections import Counter


# =====================================================
# Answer generation prompt (with retrieved documents)
# =====================================================

def _location_tag(metadata: dict) -> str:
    """Human-readable location suffix, e.g. 'p. 3' or 'part 2', if known."""
    page = metadata.get("page")
    if page is not None:
        return f" (p. {page})"

    chunk_index = metadata.get("chunk_index")
    if chunk_index is not None:
        return f" (part {chunk_index})"

    return ""


def build_context(documents: list[dict]) -> str:
    parts = []

    for index, document in enumerate(documents, start=1):
        metadata = document.get("metadata", {})
        title = metadata.get("title", "Unknown")
        location = _location_tag(metadata)
        parts.append(
            f"Document {index}\nTitle: {title}{location}\nContent:\n{document['text']}"
        )

    return "\n\n---\n\n".join(parts)


def build_prompt(context: str, question: str, language: str = "en") -> str:
    if language == "ar":
        return f"""أنت نظام استخراج معلومات متقدم. مهمتك: استخرج إجابة كاملة ومفصّلة من السياق المقدم فقط.

**قواعد الإجابة — يجب اتباعها بدقة:**

1. اقرأ السياق كاملاً قبل الكتابة.
2. السؤال قد يكون بالعربية والسياق بالإنجليزية أو العكس؛ افهم المعنى بين اللغتين.
3. السؤال قد يحتوي على أخطاء إملائية أو حروف ناقصة أو كلمات عامية؛ حاول فهم المقصود من السياق.
4. إذا وجدت معلومة مرتبطة أو قريبة جدًا من السؤال في السياق، أجب بها ولا تقل إن المعلومة غير موجودة.
5. لا تقل "المعلومة غير موجودة" إلا إذا كان السياق لا يحتوي على أي معلومة مرتبطة بالسؤال نهائيًا.
6. لا تستخدم أي معرفة خارجية خارج السياق.
7. لا تبدأ بعبارات مثل "بناءً على السياق" أو "وفقاً للمعلومات".
8. لا تكرر السؤال في الإجابة.
9. اذكر الأرقام والتواريخ والأسماء والمصطلحات الدقيقة أو التقنية كما وردت **حرفيًا** في السياق. لا تستبدل مصطلحًا محددًا مذكورًا في السياق (مثل اسم نظام، مفهوم، أو مصطلح علمي) بعبارة عامة أو مرادفة حتى لو بدت أوضح أو أسهل — إذا ذكر السياق كلمة بعينها فاستخدمها هي نفسها في إجابتك.
10. الإجابة بالعربية الواضحة، مع الحفاظ على المصطلحات الإنجليزية المهمة كما هي.
11. لا تضف أبدًا في نهاية إجابتك أي سطر أو قسم بعنوان "المصادر" أو "المصدر" أو ما شابه، ولا تسرد فيه أسماء المستندات. قائمة المصادر تُبنى تلقائيًا خارج إجابتك من بيانات الاسترجاع نفسها، وأي محاولة منك لتوليدها ستُعتبر خطأ. إن احتجت الإشارة لمستند لتوضيح المعنى داخل الجملة نفسها فاذكر عنوانه ضمن السياق العادي للجملة، لا كقائمة أو سطر منفصل.
12. اكتب الشرح مباشرة دون أي عنوان أو تسمية قسم قبله.
13. ممنوع منعًا باتًا إضافة أي مثال أو تفصيلة أو جملة توضيحية غير موجودة حرفيًا في السياق، حتى لو بدت منطقية أو متوقعة أو معقولة. لا تخترع أمثلة "افتراضية" (مثل تصرفات أشخاص أو مواقف لم يذكرها النص). أضف مثالًا فقط إذا كان مذكورًا فعليًا في السياق نفسه؛ إن لم يوجد، اكتفِ بالمعلومة الأساسية دون أي مثال ودون الإشارة إلى غيابه.
14. لا تخترع أي علاقة سببية أو رابط منطقي بين معلومتين واردتين من مصادر أو فقرات مختلفة إلا إذا كان النص نفسه يذكر هذا الرابط صراحة. إذا كانت المعلومتان منفصلتين في السياق (حتى لو بدتا متعلقتين موضوعيًا)، اعرضهما كمعلومتين منفصلتين ولا تبني بينهما جسرًا تفسيريًا من عندك.
15. أجب فقط بالاعتماد على السياق المرفق أدناه. ممنوع استخدام أي معلومة من معرفتك العامة أو التدريب المسبق، حتى لو كانت صحيحة تاريخيًا أو واقعيًا، إذا لم تكن مذكورة حرفيًا في السياق.
16. اكتب الإجابة النهائية الصحيحة مباشرة فقط. ممنوع كتابة إجابة أولية ثم تصحيحها داخل نفس الرد (مثل "لا، بل" أو "تصحيح:" أو "في الواقع"). فكّر في الإجابة الصحيحة قبل الكتابة، واكتب فقط النتيجة النهائية دون أي مسار تفكير أو تراجع ظاهر للمستخدم.

**السياق:**
{context}

**السؤال:**
{question}

**الإجابة الكاملة:**"""

    return f"""You are an advanced information extraction system. Your task is to extract a complete, well-structured answer from the provided context only.

**Answering rules — follow precisely:**

1. Read the entire context before writing.
2. The question and context may be in different languages. Understand the meaning across Arabic and English.
3. The question may contain spelling mistakes, missing letters, dialect words, or mixed Arabic/English terms. Infer the intended meaning from the context.
4. If the context contains related or very close information, answer using it. Do not say it is unavailable.
5. Only say "The information is not available in the provided documents." if the context has no related information at all.
6. Do not use external knowledge outside the context.
7. Do not open with "Based on the context" or "According to the information".
8. Do not repeat the question.
9. State numbers, dates, names, and precise/technical terms exactly as they appear in the context, verbatim. Do not replace a specific term the context uses (a named system, concept, or technical term) with a more general or "clearer" paraphrase — if the context uses a particular word, use that same word in your answer.
10. Answer in clear language, preserving important technical or foreign terms exactly as written.
11. Never add a "Sources:" or "Source:" line or section at the end of your answer, and never list document names in one. The source list is built automatically outside your answer from the actual retrieval metadata — generating one yourself is an error. If you need to reference a document to clarify meaning, name it inline as part of an ordinary sentence, never as a separate list or line.
12. Write the explanation directly, with no heading or section label in front of it.
13. Do not add any example, detail, or illustrative sentence that is not stated verbatim in the context, even if it seems plausible or expected. Never invent a hypothetical example (e.g. a made-up person's actions or scenario). Only include an example if the context itself actually contains one; otherwise just give the core information with no example and no mention that one is missing.
14. Never invent a causal or logical link between two pieces of information from different sources or sections unless the text itself states that link explicitly. If two pieces of information are separate in the context (even if topically related), present them as separate facts and do not build an explanatory bridge between them yourself.
15. Answer only using the context provided below. Do not use any information from your general/pretrained knowledge, even if it is factually or historically accurate, unless it is stated verbatim in the context.
16. Write only the correct final answer directly. Never write an initial answer and then correct it within the same reply (e.g. "no, actually" / "correction:" / "wait"). Work out the correct answer before writing, and output only the final result with no visible reasoning trail or self-correction.

**Context:**
{context}

**Question:**
{question}

**Complete Answer:**"""


def build_prompt_with_memory(context: str, question: str, language: str = "en", memory: str = "") -> str:
    """Same as `build_prompt`, with optional conversation memory prepended."""
    base = build_prompt(context, question, language)

    if not memory:
        return base

    header = (
        "**ذاكرة المحادثة (للسياق فقط، لا تكررها):**"
        if language == "ar"
        else "**Conversation memory (for context only, do not repeat it):**"
    )

    return f"{header}\n{memory}\n\n{base}"


# =====================================================
# Memory-only prompt (respond tool — no retrieved documents)
# =====================================================

def build_memory_prompt(question: str, memory: str, language: str = "en") -> str:
    if language == "ar":
        return f"""أنت مساعد محادثة ودود. أجب على رسالة المستخدم الحالية أدناه، مستعينًا بذاكرة المحادثة فقط كمرجع لفهم السياق (من نتكلم عنه، ما الذي سبق ذكره)، وليس كنص جاهز تعيد نسخه.

قواعد صارمة:
1. ممنوع نسخ أي رد سابق من "ذاكرة المحادثة" حرفيًا أو شبه حرفيًا كإجابة على الرسالة الحالية، حتى لو بدا الرد السابق قريبًا من الموضوع. رسالة المستخدم الحالية تحديدًا هي التي يجب أن تُجاب.
2. إذا كانت رسالة المستخدم توضيحًا أو تصحيحًا لسؤال سابق (مثل "قصدي X مش Y")، عالج الفرق تحديدًا في ردك — لا تكرر الإجابة القديمة كما هي.
3. إذا كانت رسالة المستخدم تطلب معلومة جديدة غير موجودة حرفيًا في ذاكرة المحادثة أدناه، لا تختلقها ولا تخمّنها — قل بوضوح إنها غير متوفرة لديك الآن.
4. ممنوع منعًا باتًا إضافة أي سطر أو قسم باسم "المصادر" أو "المصدر" في ردك؛ هذه الأداة لا تملك مصادر مستندية أصلاً.
5. إذا كانت رسالة ترحيبية أو دردشة عامة، رد بشكل طبيعي دون اختلاق معلومات.

ذاكرة المحادثة (للسياق فقط):
{memory or "لا توجد ذاكرة سابقة."}

رسالة المستخدم الحالية (هذه فقط هي المطلوب الرد عليها):
{question}

الرد:"""

    return f"""You are a friendly conversational assistant. Answer the CURRENT user message below, using
the conversation memory only as background context (who/what was discussed before) — never as
ready-made text to copy.

Strict rules:
1. Never copy a previous reply from "Conversation memory" verbatim or near-verbatim as the answer
   to the current message, even if that previous reply seems topically close. You must specifically
   answer the CURRENT message.
2. If the current message is a clarification or correction of an earlier question (e.g. "I meant X,
   not Y"), address that specific difference — do not repeat the old answer unchanged.
3. If the current message asks for new information that is not literally present in the memory
   below, do not invent or guess it — say plainly that you don't have it right now.
4. Never add a "Sources:" line or section to your reply; this tool has no documents to cite.
5. If it's a greeting or general remark, reply naturally without inventing information.

Conversation memory (background context only):
{memory or "No prior memory."}

Current user message (this is the only thing you must answer):
{question}

Reply:"""


# =====================================================
# Output cleanup
# =====================================================

# Citation labels the model sometimes self-generates even though rule 12
# forbids headings and rule 11 (see build_prompt) now explicitly forbids
# adding a sources section — the real source list is computed separately
# by build_sources() from actual retrieved metadata. If one of these
# appears, the WHOLE line is dropped (label + document names it
# introduces), never just the label — a half-strip used to leave the
# document titles stuck onto the answer as if they were prose (e.g.
# "...الرومانية المصادر الأسرات والملوك في مصر القديمة").
_BANNED_FULL_LINE_OPENERS = [
    "Sources:", "Source:", "المصادر:", "المصدر:",
]

# Sentence-starter fillers: only the matched prefix is stripped, the rest
# of the sentence is kept (unlike the citation labels above).
_BANNED_PREFIXES = [
    "Based on the context,", "According to the context,",
    "بناءً على السياق،", "وفقاً للمعلومات المتاحة،",
]

# Leading markdown/decoration (bold markers, bullets, headings, dashes)
# that can sit in front of a banned opener without defeating the match —
# a raw "**المصادر:**" line must still be caught, not silently kept
# because of the surrounding "**".
_LEADING_DECORATION = r"[\s*_>#\-•]*"

_FULL_LINE_OPENER_RE = re.compile(
    r"^" + _LEADING_DECORATION + r"(" + "|".join(re.escape(o) for o in _BANNED_FULL_LINE_OPENERS) + r")",
    re.IGNORECASE,
)

_PREFIX_OPENER_RE = re.compile(
    r"^" + _LEADING_DECORATION + r"(" + "|".join(re.escape(o) for o in _BANNED_PREFIXES) + r")",
    re.IGNORECASE,
)

NO_ANSWER = {
    "en": "The information is not available in the provided documents.",
    "ar": "المعلومة غير موجودة في المستندات المتاحة.",
}

# Safety net for rule 16 above: even with the explicit instruction, the
# model can still slip an initial (often wrong) answer followed by an
# inline correction into the SAME line/sentence (e.g. "...رمسيس الثاني
# لا، بل كليوباترا..."). clean_answer's line-level dedup below can't
# catch that — there's no line break to work with — so this strips
# everything up to and including the last correction marker found,
# keeping only the corrected tail.
_SELF_CORRECTION_MARKERS = [
    r"لا،?\s*بل\s+",
    r"تصحيح:\s*",
    r"الصحيح\s+(?:هو|أنه)\s+",
    r"في\s+الواقع،?\s*",
    r"\bno,?\s+(?:wait|actually)\b[,:]?\s*",
    r"\bactually,?\s*",
    r"\bcorrection:\s*",
]

_SELF_CORRECTION_RE = re.compile("|".join(_SELF_CORRECTION_MARKERS), re.IGNORECASE)


def _strip_self_correction(text: str) -> str:
    """Keep only the text after the LAST self-correction marker on each
    line (a line can in principle contain more than one course-correction;
    the final segment is the one the model actually intends to stand by)."""

    cleaned_lines = []

    for line in text.splitlines():
        last_match = None
        for match in _SELF_CORRECTION_RE.finditer(line):
            last_match = match
        cleaned_lines.append(line[last_match.end():].strip() if last_match else line)

    return "\n".join(cleaned_lines)


def _normalize(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().lower())


def clean_answer(text: str, language: str = "en") -> str:
    """
    Strip banned openers, visible self-correction ("no, actually..."),
    and de-duplicate repeated lines from a raw model response, matching
    the polish applied to every answer before it reaches the user.
    """
    text = (text or "").strip()
    text = _strip_self_correction(text)

    lines: list[str] = []
    seen: list[str] = []

    for line in text.splitlines():

        # Self-generated citation line (e.g. "Sources: X, Y" or a bolded
        # "**المصادر:**") — drop the ENTIRE line, not just the label, so
        # the document names it introduces don't leak into the answer as
        # if they were prose.
        if _FULL_LINE_OPENER_RE.match(line):
            continue

        prefix_match = _PREFIX_OPENER_RE.match(line)
        if prefix_match:
            line = line[prefix_match.end():].strip()

        key = _normalize(line)

        if key and key in seen:
            continue

        lines.append(line)

        if key:
            seen.append(key)
            if len(seen) > 10:
                seen.pop(0)

    result = "\n".join(lines).strip()

    return result or NO_ANSWER.get(language, NO_ANSWER["en"])


# =====================================================
# Source attribution
# =====================================================

def build_sources(documents: list[dict], language: str = "en") -> str:
    """
    Build a human-readable 'Sources:' line from retrieved documents.

    When location metadata (page number, or chunk index as a fallback) is
    available, each source is annotated like Project A's citations, e.g.
    'syllabus.pdf (p. 2, 5)'. Documents with no location concept (e.g. a
    spreadsheet) simply show the title, so citations degrade gracefully
    instead of breaking.
    """
    if not documents:
        return ""

    counts: Counter = Counter()
    titles_seen: list[str] = []
    # Per title: (locations seen, whether they are true page numbers or a
    # chunk-index fallback) so the label ("p." vs "part") stays accurate.
    locations_by_title: dict[str, list] = {}
    is_page_by_title: dict[str, bool] = {}

    for document in documents:
        metadata = document.get("metadata", {})
        title = metadata.get("title", "Unknown")
        counts[title] += 1
        if title not in titles_seen:
            titles_seen.append(title)
            locations_by_title[title] = []

        page = metadata.get("page")
        location = page if page is not None else metadata.get("chunk_index")
        is_page_by_title.setdefault(title, page is not None)

        if location is not None and location not in locations_by_title[title]:
            locations_by_title[title].append(location)

    top_titles = [title for title, _ in counts.most_common(3)]

    formatted = []
    for title in top_titles:
        locations = sorted(locations_by_title.get(title, []), key=lambda v: (isinstance(v, str), v))
        if locations:
            if is_page_by_title.get(title):
                label = "ص" if language == "ar" else "p."
            else:
                label = "جزء" if language == "ar" else "part"
            formatted.append(f"{title} ({label} {', '.join(str(p) for p in locations)})")
        else:
            formatted.append(title)

    prefix = "المصادر: " if language == "ar" else "Sources: "

    return prefix + " | ".join(formatted)