# Deep AI Chatbot & RAG Audit — Readiness for ~15,000 University Users

**Scope:** the AI/Chatbot subsystem only (`backend/agent/`, `backend/services/`, `backend/loaders/`, `backend/memory/`, `backend/routes/`, `backend/config.py`), audited by direct code tracing.
**Explicitly out of scope as "defects"** (per instructions — these belong to the production backend integration): authentication, authorization, user/account management, the production database, MinIO/object-storage architecture, general API/frontend architecture, deployment/Docker production config, and file-ownership/security model. Where any of these intersect with AI-subsystem behavior, they are called out as **integration requirements**, not defects.
**Target deployment:** ~15,000 registered university users, realistic concurrency (not 15,000 simultaneous), with attention to burst periods (registration, exams, admissions, results).

---

## 1. Executive Summary

This AI subsystem is a genuinely well-engineered prototype: a real ReAct agent (not a hardcoded retrieve→generate pipeline), a layered hybrid-retrieval stack (query expansion, cross-encoder + lexical reranking, MMR diversification, context budgeting), a well-bounded two-tier memory system that does **not** suffer unbounded token growth in long conversations, and unusually thorough in-code documentation of *why* specific engineering decisions were made — including several documented bugfixes for real, previously-observed failure modes (excessive Groq calls, planner premature-termination, conversation-id leakage).

Against a 15,000-user university deployment, three categories of finding matter most:

1. **A confirmed, still-present bug** in the semantic chunking strategy that silently discards all ingested content (not the default strategy, but a real defect).
2. **Two confirmed thread-safety race conditions** in lazy-loaded model singletons (Whisper STT and the cross-encoder reranker) that are reachable under real concurrent load — the very first few concurrent requests after a cold start can trigger duplicate, wasteful (and in the CUDA case, potentially unsafe) concurrent model loads.
3. **An architectural question the integration team must resolve before scale**: the current design isolates every uploaded document by browser-tab `conversation_id`, with no concept of a shared knowledge base. At 15,000 students, if the intent is "every student can ask about the same official university documents," the current per-conversation ingestion model means **the same PDF gets re-parsed, re-chunked, and re-embedded independently by every single student who uploads it** — a massive, avoidable multiplication of compute, with zero cross-student caching benefit, and no mechanism today to distinguish "the official, authoritative version" from a stale duplicate.

None of the "Critical" or "High" findings below are about missing auth or infrastructure — they are about correctness, concurrency-safety, grounding reliability, and cost/latency multipliers **inside the AI logic itself**, verified directly against the code that implements them.

---

## 2. Scope & Assumptions

- Every code reference below (file, function, config default) was verified against the actual source in this repository, not inferred from `README.md` or `.env.example`. Where the README's description diverges from the code, the code's actual behavior is what's reported.
- "Requires benchmarking" is used explicitly wherever a claim needs an actual measured number (throughput, latency, WER/CER, recall) that does not exist anywhere in this repository. No numbers are invented.
- Concurrency figures ("15,000 users") are treated as a target *user base*, not simultaneous load; realistic concurrency scenarios are modeled separately in Part 15 (Performance & Scalability).
- Findings are explicitly labeled as one of: **Existing bug**, **Existing weakness**, **Future/scale risk**, or **Integration requirement** — never blended.

---

## 3. Current AI Architecture

The template pipeline in the brief needs two corrections to match the actual implementation: (a) the agent's planner is a *separate*, smaller LLM from the one used for generation/expansion, and (b) `retrieve` is not a single step but can loop multiple times before a terminal tool fires.

```text
User (typed or voice)
   │
   ▼
Chat Interface (WS /ws/chat, streaming — or POST /api/chat, non-streaming)
   │
   ▼
Conversation session (agent/session.py — one in-process Agent per conversation_id,
   idle-evicted after AGENT_IDLE_TIMEOUT_SECONDS=1800s)
   │
   ▼
Agent / Planner  (agent/agent.py — ReAct loop, ≤ AGENT_MAX_ITERATIONS=6 iterations)
   │  each iteration: 1 Groq call to AGENT_MODEL (llama-3.1-8b-instant, JSON mode)
   │  chooses exactly ONE action per step, from a Pydantic-validated discriminated union
   ▼
Tools (backend/agent/tools/*.py)
   ├── retrieve   (non-terminal, can repeat) → services/rag_service.py::retrieve()
   ├── generate   (terminal)                 → rag_service.generate_answer[_stream]()
   ├── summarize  (terminal)                 → rag_service.summarize[_stream]()
   ├── compare    (terminal)                 → rag_service.compare[_stream]()
   ├── respond    (terminal, memory-only)     → rag_service.answer_from_memory[_stream]()
   └── report     (terminal, own pipeline)    → services/report_service.py (map-reduce)
   │
   ▼
RAG core (rag_service.py): query expansion → embed → Qdrant search → rerank → MMR → context budget
   │
   ▼
Memory (memory/*): short-term (RAM) + long-term FactStore (disk), injected into every prompt
   │
   ▼
Final answer (+ sources string, + optional report link) → streamed to frontend
```

### Component-by-component

| Component | What it does | Implementation | Strength | Weakness | 15K-appropriate? |
|---|---|---|---|---|---|
| Planner | Chooses next tool via structured JSON | `agent/llm.py::AgentLLM`, `AGENT_MODEL` (small/fast model) | Cheap, fast, schema-validated (Pydantic discriminated union — can't emit an invalid action) | Small model occasionally violates its own hard rules (see Part 4) | Yes, by design (cheap model specifically chosen for per-turn call volume) |
| Retrieval | Finds relevant chunks | `rag_service._retrieve()` | Multi-variant, cross-encoder+lexical hybrid, MMR-diversified | No caching, no candidate-count cap before reranking, unbounded per-request thread pools (see Part 16) | Conditionally — works, but the concurrency pattern needs review before high load |
| Generation | Produces the answer | `rag_service.generate_answer[_stream]()` | Strong, repeated, bilingual grounding rules | Grounding is 100% prompt-based, zero independent verification | Conditionally — quality is good, but unverified at scale without an eval harness (Part 20) |
| Memory | Multi-turn context | `memory/` (two-tier: RAM + disk FactStore) | Genuinely bounded — token cost does NOT grow with conversation length (Part 9) | Dedup is character-similarity, not semantic; no cross-turn semantic retrieval beyond the fact list | Yes — this is one of the stronger-engineered parts of the system |
| OCR (printed) | Extracts text from scans/images | `services/ocr_service.py` (Tesseract, multi-strategy) | Handles ara+eng jointly, multiple preprocessing passes | Very CPU-heavy per page (up to 15 Tesseract calls/image, 6/page for scanned PDFs); no quality signal captured | Needs review — real bottleneck under bulk document upload |
| OCR (handwritten) | On-demand handwriting recognition | `services/handwritten_ocr_service.py` (TrOCR ×2 models) | Automatic line segmentation (classic CV, well-engineered) | Per-line inference is strictly sequential, not batched (Part 11) | Needs a fix before heavy use |
| Voice (STT) | Speech-to-text | `services/audio_service.py` (Whisper "small") | Egyptian-Arabic-aware second pass improves accuracy | Doubles compute for Arabic audio; **unlocked lazy singleton — race condition** (Part 16) | Needs a fix before heavy use |
| Vector store | Chunk storage/search | Qdrant, one shared collection, filtered by `conversation_id` | Server mode, retried, graceful degradation | No payload-field index on `conversation_id`; single collection is an integration question at 15K scale | Integration requirement (see Part 15) |
| LLM provider | All Groq calls | `services/llm_provider.py` | Single choke point, clean abstraction, streaming support | No configured request timeout, no rate-limit-aware retry (Part 6) | No — needs hardening before production traffic |

---

## 4. Agent Architecture Audit

### 4.1 How planning works

`Agent.run()`/`run_stream()` loop (`agent/agent.py::_run_impl`/`_run_stream_impl`), up to `AGENT_MAX_ITERATIONS=6` iterations. Each iteration:

1. `_build_messages()` renders `SYSTEM_PROMPT` (fixed, `agent/prompt.py`) + a `USER_PROMPT` populated with the raw question, the "active document," rendered memory, chunk count so far, `observations` (JSON trace of every tool call this turn), and `retrieved_questions` (verbatim list of prior retrieve queries this turn).
2. `AgentLLM.invoke()` calls Groq (`AGENT_MODEL`, JSON mode), validates the response against `TypeAdapter(AgentAction)` (a Pydantic discriminated union over the 6 tools), retries up to 2 more times on invalid JSON/schema mismatch, then falls back deterministically to a `retrieve` action with the raw question if the model still can't produce valid structured output.
3. `_correct_premature_terminal()` runs (see 4.2 below) — a deterministic, no-LLM-call backstop.
4. If the action is `retrieve`: checked against `retrieved_questions` for an **exact case-insensitive string match**; if duplicate, skipped (an `"already_retrieved"` observation is appended) and the loop continues — but this still consumes one of the 6 iterations and, critically, **still costs one more planner LLM call next iteration**, since the for-loop simply `continue`s back to step 1. If not a duplicate, the tool runs, `context.retrieved_questions` grows, and `Agent.active_document` is updated from whichever source dominated the new chunks.
5. If the action is terminal (`generate`/`summarize`/`compare`/`respond`/`report`): that tool runs once and the loop breaks.
6. If all 6 iterations are exhausted without a terminal action: the non-streaming path force-calls the `generate` tool directly on whatever context exists; the streaming path calls `rag_service.generate_answer_stream()` directly (bypassing the tool wrapper entirely, so no `observations` entry is recorded for this fallback path — a minor asymmetry worth noting for observability).

**Bounded?** Yes — hard-capped at 6 iterations, always terminates with *some* answer.
**Deterministic or LLM-controlled?** LLM-controlled at the *tool-selection* level, but constrained by (a) strict schema validation, (b) the deterministic premature-terminal backstop, and (c) the exact-match duplicate-retrieve guard. This is a reasonable hybrid — not purely LLM-trusted, but not rigid either.
**Observable?** Per-iteration thought/action is logged only when `AGENT_DEBUG=true` (or `debug=True`, unused by any current caller); `context.observations` is part of the prompt (so the model *can* see its own trace) but is **not exported as a metric anywhere** — no counter for iterations-per-turn, tool-usage distribution, or duplicate-retrieve rate exists (see Part 19).
**Survives restart?** No — in-process registry (`agent/session.py::_agents`), by design; long-term facts do survive via disk (this is correct, expected behavior for the current single-process design, not a defect).
**Scales horizontally?** No — this is an **integration requirement**, not a defect: if/when the production system runs multiple backend replicas, the in-process `Agent`/short-term-memory registry must either be externalized (e.g., Redis-backed) or every conversation must be sticky-routed to the same replica for its lifetime. This needs to be decided *before* integration, since it affects the AI subsystem's own state-management contract.

### 4.2 Premature terminal behavior — deep dive

**Original failure mode** (documented in the code's own comments, referenced as "Issue 2"): the small `AGENT_MODEL` occasionally chose `respond` or `generate` on iteration 1, with nothing retrieved yet, for messages that plainly needed document lookup — most often short imperative phrasings without a question mark (e.g., "Explain X" / Arabic "اشرح لي X"), which the small model intermittently misclassified as small talk despite the system prompt's explicit "HARD RULE" against it. The code notes this was reproduced directly: *"Same exact input, temperature 0, has been observed to occasionally choose 'retrieve' and occasionally not — hosted LLM inference is not perfectly reproducible run-to-run."*

**Where:** `agent/agent.py::Agent._correct_premature_terminal()`, called immediately after every planner call in both `run()` and `run_stream()`.

**Exact mechanism:**
```python
if action.action not in (ToolName.RESPOND, ToolName.GENERATE):
    return action                                   # not applicable
if context.retrieved_questions or context.documents:
    return action                                   # already retrieved this turn — trust the model
if _looks_like_small_talk(question):
    return action                                   # genuinely matches the hardcoded phrase list
# else: force-override to RetrieveAction(question=question[:500], top_k=5)
```
`_looks_like_small_talk()` is a **fixed, hardcoded phrase set** (English + Egyptian-Arabic greetings/thanks/farewells/meta-conversation, ~45 exact strings) matched only after normalization (strip trailing punctuation, collapse whitespace, lowercase) — and **only** if the message is 6 words or fewer.

**What it correctly fixes:** the specific documented bug — a real factual/document question being answered from nothing (or from stale memory) without ever consulting the knowledge base. This backstop deterministically and cheaply (zero extra LLM calls, a design decision explicitly justified in the code as replacing a more expensive prior approach that re-asked the LLM) closes that gap for **any** iteration-1 respond/generate choice that isn't an exact match to the phrase list.

**Where it can still fail — verified, not assumed:**
1. **False "not small talk" → wasted retrieval.** Any genuine small-talk message not in the fixed phrase list (a paraphrase, a different Arabic dialect greeting, more than 6 words, an emoji-only message, "thanks a million" vs. the listed "thanks a lot") is force-routed into a real `retrieve` call — an unnecessary Qdrant query plus (if `QUERY_EXPANSION_ENABLED=true`, the default) 2 more Groq calls for query rewriting/translation. Not a correctness failure (the user still gets a reasonable answer via the `generate` tool's memory-only fallback when no documents come back), but a real, avoidable latency and Groq-cost cost on every miss. For a bilingual Egyptian-university population with wide dialectal variation, this list will be missed often.
2. **Gap the backstop does *not* cover:** it only fires "when nothing has been retrieved yet" this turn. If the planner *does* retrieve correctly on iteration 1, but then on iteration 2 — now holding real, relevant `context.documents` — chooses `respond` instead of `generate` (i.e., decides to answer from memory alone and ignore the documents it just retrieved), the guard's second condition (`if context.retrieved_questions or context.documents: return action`) means **it is trusted as-is, unchecked.** This is a distinct, uncovered premature-termination pattern: "retrieved evidence, then answered without using it." Whether the current `respond` tool's own strict memory-only prompt (§7 below) sufficiently mitigates this in practice is unverified — there is no test exercising this path.
3. **Language/dialect coverage is a closed list, not a classifier.** This is a deliberate simplicity/cost trade-off (explicitly a "no extra LLM call" design), but it means coverage will not improve as usage grows — it is not a learning or adaptive mechanism.

**Is the current solution sufficient for production at 15K users?** Partially. It correctly, cheaply, and deterministically eliminates the *dangerous* direction of the original bug (skipping retrieval for real questions) — that part is solid and should not be re-engineered. It is *not* sufficient as-is for the *efficiency* direction (over-triggering retrieval on unrecognized small talk) at scale, where every miss costs 2 extra Groq calls and a Qdrant round-trip across potentially thousands of casual "شكرا" / "تمام" / "cool thanks!" messages per day. **Recommendation:** replace the fixed phrase-list check with a cheap local heuristic informed by message length + presence of interrogative/imperative markers, or (better) a tiny local classifier — not another LLM call — to widen small-talk recall without regressing the safety property. This is a genuine, actionable, low-risk improvement, not a rewrite.

### 4.3 Pathological-loop risk (new finding, not previously documented)

The duplicate-retrieve guard is **exact-string match only** (`current_question in previous_questions`, both lowercased). A planner that rephrases its own retrieve query slightly between iterations (e.g., adding a trailing space, a synonym, or a minor reword — plausible for a small, less-reliable model under the "Multi-Question Rules" prompt section, which explicitly *encourages* issuing multiple distinct retrieve calls for sub-questions/entities) will **not** be caught as a duplicate, and will trigger a genuinely new Qdrant search plus 2 more Groq query-expansion calls, every time, until `AGENT_MAX_ITERATIONS=6` is exhausted. This is bounded (6 iterations max) so it cannot hang, but in the worst case a single confused planning sequence can cost up to **6 planner calls + up to 12 query-expansion calls + up to 6 Qdrant searches** before the max-iteration fallback forces an answer — a real, traceable cost multiplier under a bad-luck planning sequence, not a hypothetical one.

---

## 5. RAG Audit

### 5.1 Pipeline as actually implemented

```text
Question + conversation_id
   ↓
detect_language()  (character-range heuristic, ar vs en, binary — no dialect/script variant modeling)
   ↓
_query_variants(): local variants (normalized/loose forms) + 2 CONCURRENT Groq calls
   (GROQ_MODEL, the LARGE model — not AGENT_MODEL):
     - _rewrite_query(): typo-fix + up to 3 synonym/concept alternatives (one combined call, cached via
       @lru_cache(maxsize=256) keyed on exact (query, lang, max_alternatives))
     - _translate(): translate into the other language (cached via @lru_cache(maxsize=512))
   → up to 22 total variant strings
   ↓
embeddings.embed_queries(variants)   — ONE batched forward pass, local model, no cache
   ↓
concurrent Qdrant similarity_search_with_score_by_vector() per variant
   (fresh ThreadPoolExecutor per call, size = variant count — see Part 16)
   filtered server-side: metadata.conversation_id == conversation_id
   ↓
dedup (content-prefix + source + page)
   ↓
[optional source_filter — report generation only, applied client-side]
   ↓
_rerank(): cross-encoder score (ALL deduped candidates, one batched CrossEncoder.predict call,
   NO candidate-count cap) ⊕ lexical/bigram score, blended via RERANK_ALPHA=0.6
   ↓
confidence gate: reject entirely if top blended score < CONFIDENCE_THRESHOLD=0.05
   (self-documented in code as a coarse "near-zero-overlap" filter only, not a relevance classifier)
   ↓
_diversify() — MMR reselection over top max(top_n*4, 12) candidates, MMR_LAMBDA=0.7
   (only runs if candidate pool > top_n; another embedding batch call)
   ↓
top_n chunks (RERANK_TOP_N=6, widened to EXCEL_RERANK_TOP_N=12 if Excel-sourced)
   ↓
_build_context() / _trim_to_budget() — CHARACTER budget (MAX_CONTEXT_CHARS=6000), not token budget
   ↓
build_prompt_with_memory() → Groq (GROQ_MODEL) → answer
```

### 5.2 Retrieval quality

| Aspect | Finding |
|---|---|
| Recall/Precision/MRR/nDCG | **Not measured anywhere in the repository.** No eval dataset, no metric computation. Every design choice below (variant count, top-k, rerank blend weight, MMR λ) is a reasoned default, not a validated-optimal one. See Part 20. |
| Top-K selection | `RETRIEVER_K=8` per variant → potentially large raw candidate pool before dedup (up to 8×22=176 in the theoretical worst case, realistically much lower due to overlap) → deduped → reranked down to `RERANK_TOP_N=6`. Reasonable shape, unvalidated tuning. |
| Duplicate results | Handled at two levels: `_deduplicate_retrieved` (content-prefix+source+page) after retrieval, and MMR (semantic near-duplicate suppression) at selection time. Reasonably layered. |
| Chunk relevance | Enforced only by the coarse confidence threshold + rerank ordering — no hard relevance floor beyond the top score. |
| Metadata/conversation/document filtering | `conversation_id` filtering is correctly and consistently applied inside `_retrieve()` — this specific mechanism is sound (the isolation weaknesses documented in the prior report live in *other*, adjacent endpoints, not here). `source_filter` (document-level) exists but is applied **client-side after** the Qdrant query, not as a server-side filter — meaning a topic-scoped, single-document report request still fetches `TOPIC_REPORT_TOP_K=24` candidates across the *whole conversation's* knowledge base from Qdrant before filtering down client-side to the named document — a minor but real inefficiency at scale (over-fetching from Qdrant when a server-side `source` filter would be cheaper and more precise). |
| Cross-document retrieval | Works as designed — a single retrieve call can and does surface chunks from multiple documents; nothing structurally prevents this. |
| Multilingual retrieval | Addressed via query-variant translation + the multilingual embedding/reranker models — genuinely reasoned design, unverified quality (Part 10). |

### 5.3 Query expansion

- **Model used:** the **large** `GROQ_MODEL` (default `llama-3.3-70b-versatile`), not the cheaper `AGENT_MODEL`. This is a deliberate cost/latency inconsistency worth flagging: the planner already uses a cheaper model for its (arguably harder) structured-reasoning task, but query rewriting/translation — a comparatively simpler, more mechanical NLP task — always uses the expensive model. **Recommendation:** evaluate whether `AGENT_MODEL` (or an even smaller/faster model) produces acceptably similar rewrite/translation quality; this single change could meaningfully cut per-turn LLM cost and latency, since this call fires on **every single retrieve step**, including every iteration of a multi-entity comparison.
- **Variant count:** up to 22, capped explicitly in code (`variants[:22]`).
- **Does it improve recall?** Plausible by design (this is its stated purpose), but **entirely unmeasured** — no A/B test, no recall metric exists.
- **Query drift risk:** real — LLM-generated synonym/concept alternatives (e.g., "advantages" → "benefits") can occasionally introduce semantically loose variants that pull in tangential content, partially mitigated (not eliminated) by downstream reranking.
- **Cost:** 2 concurrent Groq calls per `retrieve` invocation (when `QUERY_EXPANSION_ENABLED=true`, the default), each on the largest/most-expensive model — this is the single most repeatable, highest-volume LLM cost driver in the whole system (see Part 6 table).
- **Needed for every query?** No — a cheap query already sharing exact vocabulary with the documents (e.g., an exact course-code lookup) gains little from LLM-based expansion; a pre-check that gates expansion behind a fast heuristic (e.g., skip expansion for very short, keyword-like queries with a strong lexical match already found) is a viable, unimplemented cost optimization.

### 5.4 Reranking

- **Model:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` — compact, multilingual, CPU-reasonable choice.
- **Candidate count:** **no explicit cap** — `_rerank()` scores every deduped candidate from `docs` in one batched `predict()` call. Bounded in practice by `RETRIEVER_K × variant count` before dedup, but there is no independent safety ceiling on cross-encoder batch size — a document set with high semantic overlap across many query variants (plausible for a dense, repetitive official document like a student handbook) could produce a larger-than-typical batch with no explicit limit.
- **Latency:** code comments in `utils/device.py` document real, measured figures for a *related* operation (the MMR embedding step, ~24 passages): ~1.7–1.9s CPU / ~0.17s GPU (~10x), and separately ~160ms CPU / ~25–30ms GPU "warm" for the cross-encoder — these are the only real numbers in the repository; the candidate count they correspond to is not stated precisely, so extrapolating to production-scale candidate counts **requires benchmarking**.
- **Runs on every query:** yes, unconditionally (when enabled, the default), with a documented permanent fallback to lexical-only scoring if the model fails to load even once in the process lifetime.

### 5.5 Lexical scoring

Exists as a fallback and a blend partner (`RERANK_ALPHA=0.6` favors the cross-encoder 60/40). This is **more valuable than in a generic RAG system** for this specific university use case: student questions frequently target exact identifiers (course codes, GPA figures, specific deadlines, form numbers) that benefit disproportionately from exact/bigram substring matching alongside semantic similarity, since a purely semantic ranker can under-weight an exact numeric/code match in favor of topically-similar-but-wrong content. This is a genuine strength of the current design for this domain, worth preserving.

### 5.6 MMR

`MMR_LAMBDA=0.7` (70% relevance weight, 30% diversity). Real risk, not hypothetical: if the top few genuinely-best chunks are also mutually similar (e.g., the same fact repeated across several sections of a syllabus, or multiple near-identical PDF copies of the same official document uploaded separately — see §5.9), MMR can push one of the best chunks out in favor of a more diverse but less relevant one. For *factual* QA, redundant confirmation across chunks is often harmless or even beneficial (higher confidence the fact is real), so diversity is not an unambiguous win here — this specific parameter (λ=0.7) should be validated empirically (Part 20) rather than trusted as correctly tuned.

### 5.7 Context budgeting

- `MAX_CONTEXT_CHARS=6000` is a **character** budget, not a token budget. In a bilingual Arabic/English system, this is a real, code-verifiable inconsistency: Arabic UTF-8/BPE tokenization is generally less token-efficient per character than English (more tokens per character for the same visible text length, depending on the tokenizer), so a fixed 6000-character cap does not translate to a consistent token budget across languages — an Arabic-heavy context can consume meaningfully more of the model's context window than an English one of identical character length. **Recommendation:** switch to a token-aware budget (a lightweight tokenizer-based count, or at minimum a language-aware character multiplier) rather than a flat character count.
- **Interaction with chunk sizing:** `_trim_to_budget()` operates *after* rerank/MMR already selected `top_n=6` chunks — not on the full candidate pool. With `hybrid`/`semantic`-strategy chunks capped at up to 1800 characters each, 6 selected chunks can total up to 10,800 characters — comfortably exceeding the 6,000-character budget — meaning the trim step will **routinely discard 2–3 of the 6 chunks the retrieval pipeline worked to rank and select**, silently, with the LLM seeing fewer distinct pieces of evidence than the pipeline actually found. This is a real, traceable tension between `HYBRID_CHUNK_MAX_CHARS`/`SEMANTIC_CHUNK_MAX_CHARS` and `MAX_CONTEXT_CHARS` that is not addressed anywhere in the tuning.

### 5.8 Grounding via prompt vs. code

See Part 7 for the full hallucination/grounding classification.

### 5.9 A structural RAG design question the integration team must resolve

`rag_service._registry_key()`'s docstring is explicit that dedup is **deliberately scoped per-conversation, not globally** — a design decision that made sense for a per-tab prototype where "conversation" ≈ "one user's private session." At 15,000 students, if the intended production model is "every student can ask about the same official university handbook/regulations," this per-conversation model means:

- The same official PDF gets independently re-parsed, re-OCR'd (if scanned), re-chunked, and re-embedded **every single time a different student uploads it** — no sharing, no caching benefit across the 15,000-user base for what is very likely the *most commonly asked-about content*.
- There is no concept of an "authoritative, shared knowledge base" distinct from "a student's own private uploads" anywhere in the current data model (`conversation_id` is the only scoping key that exists).
- This directly limits the biggest possible win identified in Part 14 (Caching): cross-user answer/retrieval caching for common FAQ-style questions is not meaningfully possible under the current per-conversation isolation model, because each student's "knowledge base" is logically a separate silo even when the underlying documents are byte-identical.

**This is not a bug in the current prototype** — it is a legitimate, documented design decision for its original scope — but it is the single most consequential **integration requirement** for a 15,000-user university deployment, and should be decided explicitly (shared global knowledge base + optional per-student private uploads, vs. continuing pure per-conversation isolation) before significant further AI-subsystem work is invested in caching or scaling around the current model.

---

## 6. LLM Audit

### 6.1 Call inventory

| Operation | Model | Approx. calls / occurrence | Input | Output | Risk |
|---|---|---:|---|---|---|
| Agent planning (per iteration) | `AGENT_MODEL` (llama-3.1-8b-instant) | 1 per iteration, ≤6/turn | System + user prompt (question, memory, observations) | JSON action | Schema-validated; falls back deterministically on failure |
| Query rewrite + synonyms | `GROQ_MODEL` (llama-3.3-70b-versatile) | 1 per `retrieve` call (cached by exact query text) | Question | JSON `{corrected, alternatives}` | No rate-limit handling; largest-model cost for a mechanical task |
| Query translation | `GROQ_MODEL` | 1 per `retrieve` call (cached) | Question | Translated text | Same as above |
| Final generation | `GROQ_MODEL` | 1 per terminal `generate`/`respond` | Context + memory + question | Free-text answer | No independent verification (Part 7) |
| Summarize | `GROQ_MODEL` | 1 per terminal `summarize` | All retrieved chunks | Free-text summary | — |
| Compare | `GROQ_MODEL` | 1 per terminal `compare` | All retrieved chunks (multi-entity, ungrouped — see Part 13) | Free-text comparison | Prompt does not explicitly separate sources |
| Memory fact extraction | `GROQ_MODEL` (via `LLMTextGenerator`) | 1 per ~21 messages (background, non-blocking) | Recent messages + existing facts | JSON `{facts, remove}` | Unbounded concurrent background threads at scale (Part 16) |
| Report generation — MAP step | `GROQ_MODEL` | 1 per ~6,000-char document slice, up to `MAP_EXTRACT_CONCURRENCY=5` concurrent | Document slice | Structured JSON extraction | Bounded concurrency (good) |
| Report generation — REDUCE step | `GROQ_MODEL` | 4 (executive summary, introduction, relationships, conclusion), concurrent | Aggregated facts/digest | Free text | Bounded (good) |
| Report topic-relevance gate | `GROQ_MODEL` | 1 per topic-scoped report request | Sample of retrieved chunks + topic | "yes"/"no" | Fails open (treats as relevant on error) |

### 6.2 Realistic per-turn call volume

```text
Simple single-lookup question (1 retrieve → generate):
  1 planning call (iter 1) + 2 query-expansion calls (rewrite+translate, concurrent)
  + 1 planning call (iter 2, terminal) + 1 generation call
  = 5 Groq API calls per user turn (3 network round-trips of wall time, since 2 are concurrent)

Two-entity comparison (2 retrieves → compare):
  ≈ 3 planning calls + 4 query-expansion calls + 1 comparison call = 8 Groq API calls

Report generation (whole document, ~5 slices):
  ≈ 5 MAP calls (concurrency-bounded to 5 at once) + 4 REDUCE calls ≈ 9 Groq API calls,
  on top of whatever agent planning got it there (report is chosen directly, no retrieve needed)

Worst-case pathological planning sequence (Part 4.3): up to 6 planning + 12 expansion calls
  before the max-iteration fallback forces an answer ≈ 18-19 Groq calls for one user turn
```

Scaling this linearly (not accounting for any provider-side batching, since none exists):

```text
100 concurrent simple questions  ≈ 500 Groq calls in flight
500 concurrent simple questions  ≈ 2,500 Groq calls in flight
1,000 concurrent simple questions ≈ 5,000 Groq calls in flight
```

**External provider limits (Groq API rate limits, concurrent-request caps, tokens/minute) are not available anywhere in this repository and are not invented here.** This is explicitly marked: **"External provider limit — requires verification with Groq directly, and must be sized against the peak-concurrency scenarios in Part 15."**

### 6.3 Retry, backoff, timeout, fallback

- **Retry:** only the agent's planning call retries (2 extra attempts on malformed JSON, `agent/llm.py`). No retry exists for query-rewrite, translation, generation, summarize, compare, memory-extraction, or report calls — a single Groq exception on any of these propagates and is caught only at the outermost level, producing a descriptive error string instead of the intended output.
- **Exponential backoff:** does not exist anywhere for Groq calls.
- **429 (rate limit) handling:** **not implemented at all.** A rate-limited response is treated identically to any other exception — no special detection, no wait-and-retry, no queuing.
- **Timeout:** no explicit request timeout is passed to the `groq` SDK anywhere (`GroqLLM.chat`/`stream_chat` call `client.chat.completions.create(**kwargs)` with no `timeout=`) — behavior under a slow/hanging Groq response depends entirely on the SDK's own undocumented-in-this-repo default.
- **Fallback model:** none — a single hardcoded model per role (`GROQ_MODEL`, `AGENT_MODEL`); no automatic degradation to a smaller/different model or provider on sustained failure.
- **Provider outage:** every LLM-dependent turn would fail with a generic error message to the user; there is no circuit breaker, no cached/canned fallback response, no "the AI assistant is temporarily unavailable" graceful state distinct from a normal error.

This is the single highest-priority **AI reliability gap** for a production deployment expected to see genuine burst traffic (exam periods, results day) — see Part 23/24.

---

## 7. Hallucination & Grounding Audit

### 7.1 What exists

- Extensive, repeated, explicitly-worded grounding rules in `build_prompt()` (both languages): answer **only** if the context *specifically and directly* covers the question (not "a related topic"); say the fixed "not available" sentence otherwise; never use outside knowledge, even for trivially-known facts; never fabricate numbers/names; preserve equations/units verbatim; match answer length to question complexity.
- A separate, stricter prompt (`_memory_only_prompt`, used by `respond`) explicitly forbids answering *any* factual question from the model's own training knowledge, restricting it to greetings/small-talk/meta-conversation only.
- The planner's "HARD RULE" forces `retrieve` before `generate` for essentially all factual questions, backstopped in code (Part 4.2).
- A coarse pre-filter (`CONFIDENCE_THRESHOLD=0.05`) rejects near-zero-overlap retrievals outright — self-documented in the code as *not* a reliable topic classifier, only a catch for the most extreme non-matches.
- Pydantic schema validation prevents the *planner* from being hijacked into an unauthorized action (a real, structural defense — see Part 17).

### 7.2 What does not exist

- **Zero independent verification** of the generated answer against the retrieved context — no entailment/NLI check, no citation-level faithfulness check, no post-hoc "does this claim actually appear in the context" pass. Grounding is **entirely prompt-based**.
- **No contradiction handling.** If two retrieved chunks disagree, there is no instruction telling the model how to handle it (only an Excel-specific instruction exists for aggregating rows), and no code-level detection that conflicting evidence was retrieved.
- **No document versioning/staleness signal reaches the model.** This is the most consequential grounding gap for this specific domain: re-uploading an updated version of a document does **not** remove or supersede the old version's chunks (confirmed: `update_db_files()` only skips re-indexing on byte-identical content; a modified file is indexed as fully additional, independent chunks under a new `document_id`). The chunk metadata **does** capture an ingestion `timestamp` (`loaders/base.py::make_meta`), but this field is **never surfaced to the LLM** — `_chunk_label()` only shows `source` and `page`/sheet info in the prompt header, never `timestamp`. Concretely: if an updated admission-requirements PDF is uploaded to "replace" an outdated one, both versions' chunks remain permanently retrievable and indistinguishable by recency to the model, which has no basis to prefer the newer one and no instruction to flag the conflict to the user. **For a university deployment, this is a real, plausible path to a student receiving outdated official information with no visible warning — the highest-severity grounding finding in this audit.**
- **Sources ≠ verified support.** The "Sources:" line is computed from the same retrieved-chunk metadata already used to build context, independently of whether the model's specific claims are actually traceable to those chunks — it indicates "what was retrieved," not "what was actually used/faithful."
- **Ambiguity is resolved by guessing, not asking**, by deliberate design (the planner's HARD RULE strongly discourages asking for clarification before at least attempting retrieval). This is a reasonable UX trade-off in general, but it means a genuinely ambiguous question that happens to retrieve marginally-relevant chunks (passing the weak confidence gate) will receive a confident-sounding, specific answer rather than a request for clarification.

### 7.3 Classification

**Overall: Moderate**, with one **High-severity, domain-specific gap** (document staleness/versioning) that pulls the practical risk profile toward Weak for exactly the kind of regulatory/administrative content a university system will serve most. The prompt engineering itself is above-average for a project of this scope — the gap is the complete absence of any code-level verification or recency awareness layered on top of it. **This should be treated as a Phase-1/Phase-2 priority, not deferred**, given the real-world cost of a student acting on stale "official" information.

---

## 8. Memory Audit

### 8.1 What's stored, when, why

| Layer | Contents | Trigger | Persistence |
|---|---|---|---|
| Short-term (`ShortMemory`) | Raw `{role, content}` messages | Every turn | RAM only, per-Agent |
| Long-term (`FactStore`) | `{text, category, importance, updated_at}` facts | Background, when short-term crosses `MEMORY_MAX_MESSAGES=25` messages **or** `MEMORY_MAX_CHARS=12000` chars, whichever first | Disk (`memory_storage/<conversation_id>.json`), reloaded on next Agent creation |

- **Deduplication:** character-level `difflib.SequenceMatcher` ratio ≥ 0.92 — a crude but deliberately-tuned (per code comments, to avoid merging genuinely-different short facts that share boilerplate) heuristic. It is **not semantic**: two facts that are conceptually identical but phrased very differently (plausible across a code-switched Arabic/English conversation) will both be stored as distinct facts, gradually consuming the `MEMORY_MAX_FACTS=40` budget with near-duplicates the string-similarity check cannot catch. Given an embedding model is already loaded for other purposes in this same process, a semantic-similarity dedup check would be a natural, low-effort upgrade.
- **Importance ranking / eviction:** lowest `importance` (1–5, LLM-assigned) then oldest `updated_at` evicted first once the 40-fact cap is exceeded — reasonable.
- **Isolation:** per `conversation_id`, matching the same scoping used everywhere else in this subsystem — consistent, no cross-conversation memory bleed observed in the code.
- **Memory poisoning risk:** the fact-extraction prompt (`memory/fact_extractor.py::build_prompt`) has **no grounding requirement** distinguishing "stated by the user in chat" from "verified from a retrieved document" — a user (or, in principle, injected document text later quoted back into the transcript by a generated answer) can plant a false claim that the LLM may then store as a persistent "fact," which is re-injected into **every future prompt for that same conversation** until evicted or contradicted. Scoped to the single conversation only (no cross-user impact given existing isolation), but real within it. See Part 17.

### 8.2 Will this design work at 15,000 users with many/long conversations?

**Yes, with caveats.** The architecture is already well-bounded:

- **RAM growth:** bounded per-conversation (short-term buffer capped at 25 messages/12,000 chars before folding), and the whole in-process `Agent` registry is actively evicted (idle timeout 30 min, cleanup every 5 min) — this specific mechanism should hold up fine at 15K *registered* users, since only *actively conversing* users occupy RAM at any moment.
- **Disk growth:** one small JSON file per conversation, capped at 40 facts each — negligible per-file size; the *number* of files (potentially tens of thousands over time, one per conversation ever created, since old files are never garbage-collected except on explicit reset) is a long-run filesystem-scale consideration, not a memory-pressure one.
- **JSON scalability / locking / race conditions:** each conversation's fact file is independent (keyed by sanitized `conversation_id`), so **concurrent writes across different conversations do not collide** — this is a materially better situation than the single shared `processed_files.json` registry (a genuinely separate, higher-risk concern — see Part 16). Per-file writes are not atomic (`json.dump` directly to the target path, no temp-file-then-rename), so a crash mid-write to one conversation's own file could corrupt *that one file* — a narrow, low-probability risk, not a cross-conversation one.
- **Multi-instance problems:** as noted in Part 4.1, this is an **integration requirement**, not a current defect — the in-process `Agent` registry (short-term memory) does not survive across replicas; the on-disk `FactStore` files *would* survive and be re-loadable by any replica if the disk is shared (e.g., a network volume), which is a relevant design fact for the integration team to know now.

---

## 9. Long Conversation Audit

Conceptual behavior at 10 / 50 / 100 / 500 messages, traced directly from `MemoryManager.as_prompt_text()`:

```python
parts = [rendered_facts (≤ MEMORY_SUMMARY_MAX_CHARS=1200 chars, ≤ MEMORY_MAX_FACTS=40 facts)]
      + [last MEMORY_WINDOW=6 raw messages]
```

**The system does NOT send the entire history to the LLM at any conversation length.** Because summarization triggers well before the 25-message/12,000-char ceiling and repeatedly re-triggers as the conversation grows (folding down to `MEMORY_KEEP_RECENT=4` messages each time), the memory portion of every prompt stays **roughly constant-sized regardless of total conversation length** — the last 6 raw messages plus at most ~1,200 characters of facts. This is a genuine, verified architectural strength: a 500-message conversation's memory payload is **not** meaningfully larger than a 10-message one's. Token/cost growth for the memory component is flat, not linear — this directly satisfies the brief's own suggested mitigations (sliding window, summarization) **because they are already implemented**, not because they're missing.

What *does* scale with conversation length: the number of background fact-extraction LLM calls over the conversation's lifetime (~1 per 21 new messages — e.g., ≈23 such calls over a 500-message conversation), each a non-blocking background call that doesn't add to any single turn's latency but does add to cumulative Groq usage.

**What's genuinely missing (a real, if lower-priority, gap):** anything outside the last 6 raw messages is only recoverable if it was distilled into a "fact" — a message referencing something raised 30 turns ago that the extraction step didn't deem important enough to keep is **permanently inaccessible** to future turns. For long, exploratory academic-advising-style conversations, a semantic retrieval pass over the *full* message history (not just the fact list) would close this gap, but this is a nice-to-have enhancement, not an urgent fix — the current design already meets the brief's core concern (bounded cost/latency growth).

---

## 10. Arabic/English Audit

Traced against the brief's own example phrases:

| Example | Detected as | Pipeline behavior |
|---|---|---|
| "ايه شروط التقديم؟" (Egyptian dialect) | `ar` (Arabic-char-majority heuristic in `detect_language()`) | Rewrite/synonym step runs in Arabic; may reformulate dialect phrasing toward document-style wording (a genuine, useful bridge mechanism) |
| "ما هي شروط القبول؟" (Formal Arabic) | `ar` | Same pipeline — **the system does not distinguish dialect from formal Arabic anywhere**; both are handled identically as "ar" |
| "What are the admission requirements?" | `en` | Translated to Arabic as one retrieval variant, enabling a match against an Arabic-only source document |
| "عايز أعرف الـcredit hours بتاعة CS" (code-switched) | `ar` (Arabic char count dominates a short sentence with 2 embedded English terms) | Rewrite/translation run on the full mixed string; final-answer prompt rule 10 explicitly instructs the model to answer in Arabic while **preserving English technical terms** — a prompt rule directly anticipating exactly this pattern |

**Positive, code-verified findings:**
- `detect_language()` is a simple, binary Arabic-char-vs-Latin-char count — functional but not a true language/dialect identifier; this is a reasonable, cheap choice for routing purposes, not a weakness in itself.
- Tesseract OCR is correctly configured with **both** language packs simultaneously (`-l ara+eng`), the right choice for documents that mix Arabic prose with English technical terms/course codes — a deliberate, correct configuration already in place.
- The final-generation prompt (rule 10, both languages) explicitly handles code-switching: answer in the user's question language regardless of source-document language, while preserving important English/Arabic technical terms as needed — directly matches the brief's own "credit hours بتاعة CS" example.
- Arabic-specific normalization (`_normalize_arabic`: diacritic stripping, letter unification for أ/إ/آ/ا, ى/ي, etc.) is applied both at ingestion (`_enrich`) and query time — real, working, non-trivial Arabic text engineering, not a superficial label.

**Where quality is genuinely unverified (must be flagged, not assumed):**
- **Embedding model** (`intfloat/multilingual-e5-large`): a strong general multilingual model on paper, but its performance specifically on Egyptian-dialect or code-switched Arabic academic text is **not benchmarked anywhere in this repository.**
- **Reranker** (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`): trained on mMARCO, whose Arabic portion is **machine-translated, formal-register Arabic** (translated from English MS MARCO queries) — this training distribution plausibly under-represents naturally-occurring Egyptian dialect and code-switching specifically, a reasoned (not invented) quality risk that should be a first-class test category, not an afterthought, in any evaluation dataset (Part 20).
- **LLM's own dialect/code-switch understanding** (Groq-hosted Llama 3.3/3.1): generally strong for modern large models, but **unverified against this specific domain** (Egyptian university administrative/academic Arabic) in this repository.
- **Voice input asymmetry:** the Egyptian-Arabic-tuned Whisper second pass (`audio_service.py::transcribe_audio`) **doubles STT compute** for any Arabic voice message relative to English (two full model passes vs. one) — a deliberate, reasoned accuracy trade-off, but one that means Arabic voice-input latency is roughly double English voice-input latency by design, worth monitoring explicitly (see Part 12).

**Recommendation:** the evaluation dataset in Part 20 must include, as explicit first-class categories: Egyptian dialect, Formal Arabic, English, and Arabic/English code-switching — with separate scoring per category, not a single blended "Arabic" bucket — precisely because the brief's own examples show these are pipeline-relevant distinctions the current system does not model separately anywhere.

---

## 11. OCR Audit

### 11.1 Printed text (Tesseract, `services/ocr_service.py`)

- Automatic, embedded in the upload pipeline for scanned PDFs (triggered when `PyPDFLoader` extracts under 20 characters of text) and unconditionally for all image uploads.
- **Multi-strategy, multi-PSM, brute-force approach:** images run through 5 preprocessing strategies (`adaptive`, `otsu`, `denoise`, `sharpen`, `contrast`) × 3 PSM modes (6, 3, 11) = **up to 15 Tesseract invocations per image**; scanned PDF pages run 3 strategies × 2 PSM modes = **up to 6 Tesseract invocations per page**. Results are merged/deduplicated by longest-result-first, line-level dedup. This is a genuinely thorough approach to maximizing OCR accuracy on noisy scans — a real strength — but it is correspondingly **CPU-heavy per document**, with cost scaling linearly in page/image count and multiplicatively in the strategy×PSM grid.
- **No OCR quality/confidence signal is captured or surfaced anywhere.** `pytesseract.image_to_string()` (used throughout, not `image_to_data()`, which *would* return per-word confidence) means there is no way — at ingestion time, in logs, or in the upload-status response — to know whether a given document's OCR succeeded well or poorly. The chain the brief asks about is real and directly traceable in this codebase:

```text
Bad scan / low-quality photo
   ↓  (Tesseract runs all 15 strategy/PSM combinations regardless — no early "this looks unreadable" signal)
Bad/garbled text (silently accepted if ≥15 chars and contains a letter — _is_meaningful())
   ↓
Bad chunks (chunker has no awareness of OCR quality; garbled text is chunked exactly like clean text)
   ↓
Bad embeddings (the embedding model faithfully encodes whatever text it's given, garbled or not)
   ↓
Bad retrieval (a garbled chunk can still score well lexically/semantically against an unrelated query
   by coincidence, or simply never surface for the query it should answer)
   ↓
Bad or missing answer, with no signal to the user or to logs that the ROOT CAUSE was OCR quality
   rather than "the document doesn't cover this," "the model made a mistake," or any other cause
```

- **Recommendation (measurable):** capture a per-page/per-image confidence proxy (e.g., `pytesseract.image_to_data()`'s mean word confidence, or simply the character count returned per strategy as a weak proxy already computed in `_merge_ocr_results`) and surface it in the upload-status response and/or as chunk metadata, so low-confidence OCR content can at minimum be flagged to the uploader ("this scan may have been hard to read — please verify") rather than silently entering the knowledge base indistinguishably from clean text. CER/WER against a labeled sample of real university-document scans is the correct formal metric, but **requires a benchmark dataset that does not currently exist**.

### 11.2 Handwritten (TrOCR, `services/handwritten_ocr_service.py`)

- Two separate, correctly-implemented, thread-safe-loaded models (Arabic/English), with genuinely well-engineered automatic line segmentation (gradient/Otsu-based ink-density projection, not a naive intensity threshold — the code comments document real, specific reasoning for this choice, verified against actual test images).
- **Confirmed performance finding: per-line recognition is strictly sequential, not batched.** `recognize_with_debug()`:
  ```python
  texts = [self._recognize_image(img, language) for img in lines]
  ```
  A plain Python list comprehension — each detected line runs its own full forward pass (up to `HANDWRITTEN_OCR_MAX_NEW_TOKENS=256` generated tokens) through a ~334M-parameter `VisionEncoderDecoderModel`, one at a time, even though `transformers`/TrOCR natively supports batched image inputs. For a full handwritten page with many lines (bounded at `_LINE_MAX_COUNT=80`), this means up to 80 sequential model calls for a single OCR request — a real, code-verified, and easily fixable latency issue (batch the cropped line images through the model together, or at minimum process them on a small bounded thread/process pool) that will matter directly if this feature sees meaningful use (e.g., students submitting photographed handwritten notes/forms).
- No CER/WER evaluation exists anywhere in the repository for either language.

---

## 12. Voice Audit

`services/audio_service.py`, Whisper `"small"` model (default `WHISPER_MODEL_NAME`).

- **Arabic handling:** auto-detects language via Whisper's own `detect_language()`; if Arabic is detected, runs a **second, Egyptian-dialect-prompted pass** and keeps whichever transcript is more complete (length-based heuristic: keeps the second pass only if it's at least 60% as long as the first). This is a genuinely thoughtful, deliberate design for exactly the target user base — but it means **Arabic voice messages cost roughly 2x the Whisper compute of English ones**, a real, code-verified asymmetry worth monitoring explicitly at scale, not treated as free.
- **Noise handling:** only a total-silence gate (`SILENCE_THRESHOLD_DB=-60.0`, ffmpeg `volumedetect` mean-volume check) — this catches genuinely empty/silent recordings but does **not** gate on general noise quality; a noisy-but-non-silent recording (background chatter, wind, low-quality phone mic — realistic for students recording on the go) passes through unfiltered into Whisper with no pre-filtering or post-hoc confidence check.
- **Long audio:** no explicit duration cap is enforced in this module (only the general `MAX_UPLOAD_SIZE_MB` file-size limit applies, and that's actually only checked in `routes/chat.py::chat_voice`'s multipart handling indirectly via FastAPI's own body-size limits, not an explicit duration check) — a long recording simply takes proportionally longer to transcribe, with no separate timeout specific to voice processing.
- **Code switching:** relies entirely on Whisper's own capability; not specially handled (no forced bilingual decoding pass analogous to the Egyptian-Arabic second pass).
- **CPU/GPU cost, latency:** **no measured numbers exist in this repository** for the "small" Whisper model's real-world latency on representative Egyptian-Arabic university audio. **Requires benchmarking.**
- **Concurrency safety — confirmed defect (see Part 16):** `_get_model()` has **no lock** around its lazy singleton initialization, unlike the correctly double-checked-locked pattern used for the handwritten-OCR models in the same codebase. This is directly verifiable:
  ```python
  # backend/services/audio_service.py
  _model: "whisper.Whisper | None" = None
  def _get_model() -> "whisper.Whisper":
      global _model
      if _model is None:
          log.info(f"Loading Whisper model ({settings.WHISPER_MODEL_NAME})...")
          _model = whisper.load_model(settings.WHISPER_MODEL_NAME)
      return _model
  ```
  Two or more concurrent voice requests arriving on a freshly-started backend (a plausible real event — the very first users of the day, or after a restart during a burst period) can both observe `_model is None` simultaneously and both call `whisper.load_model()` concurrently — wasteful at minimum (duplicate model load, duplicate memory), and a genuine risk of undefined behavior in the worst case depending on the underlying library/CUDA context's own thread-safety during initialization. **This should be fixed before any concurrent-user testing**, and is a small, low-risk, well-scoped fix (mirror the existing lock pattern already correctly used in `handwritten_ocr_service.py`).

**Recommended evaluation approach:** build a small labeled test set of real (or realistic) Egyptian-Arabic and English voice clips relevant to the university domain (course names, admin terms, mixed-language questions), and measure WER per language/dialect category, plus end-to-end latency (including the Arabic double-pass), before relying on voice input at any real scale.

---

## 13. Multi-Document Reasoning Audit

- **Single document:** works as expected — retrieval naturally concentrates on the one relevant source.
- **Multiple documents:** works structurally — a single `retrieve` call can surface chunks from multiple sources in one pass; the planner's "Multi-Question Rules" explicitly instruct issuing **separate** `retrieve` calls per distinct entity for comparison-style questions.
- **Comparing documents:** the `compare` tool concatenates **all** accumulated chunks (potentially from several separate `retrieve` calls across different entities) into **one flat `document_text` blob**, with only inline per-chunk `[Chunk N | source | page]` headers distinguishing origin — there is **no explicit grouping/labeling by source document** in the prompt structure itself (e.g., no "=== Document: A.pdf ===" section headers). After MMR/reranking, chunks from different documents can be **interleaved** in relevance order rather than grouped by source. This relies entirely on the LLM's own ability to parse and track per-chunk source attribution from inline headers scattered through a single concatenated block — plausible for a large capable model, but not structurally guaranteed by the code. **Recommendation:** explicitly group/label chunks by source document in the `compare`/`generate`-for-comparison prompt construction — a low-difficulty, likely-meaningful quality improvement.
- **Contradictory documents / same topic across documents / different versions of the same document:** this is the same finding as Part 7's staleness gap — the system has **no mechanism to distinguish an authoritative/current version from a superseded one**, and no explicit "if sources disagree, say so" instruction exists in any prompt. Metadata that *could* help (`timestamp`) is captured but never surfaced to the model.
- **Whether metadata is sufficient to distinguish Document A vs. B:** `source` (filename) and `document_id` are always present and shown in the prompt header, which is sufficient for the model to *attribute* a chunk to its file — but insufficient to *rank* or *reconcile* conflicting files, since no recency/authority signal reaches the prompt.

---

## 14. Performance & Scalability Audit — Realistic Scenarios for 15,000 Users

Per the brief's explicit instruction, this models **realistic concurrency**, not 15,000 simultaneous users.

### Scenario A — 50 concurrent users (typical daytime load)

- Groq calls in flight: ~250 (using the 5-calls/simple-turn estimate from Part 6).
- Local ML load: ~50 concurrent embedding-batch + reranking + MMR-embedding sequences, each ~1-4 seconds of CPU compute per the only real benchmark data available (Part 5.4/`utils/device.py`), competing for whatever CPU cores the single backend process has.
- Qdrant: 50× concurrent filtered searches across up to 22 variants each — well within a single Qdrant node's typical capability for this scale, but **unbenchmarked** in this repo.
- **Likely bottleneck: none critical at this scale**, assuming reasonable hardware (4+ cores). This should work today essentially as-is.

### Scenario B — 200 concurrent users

- Groq calls in flight: ~1,000. **This is the first scenario where the complete absence of rate-limit handling (Part 6.3) becomes a real, not theoretical, risk** — if Groq's actual account-level rate limits (unknown from this repo, **requires verification with Groq**) are anywhere near this volume, a fraction of these 1,000 calls will start failing with 429s, each surfacing as a raw user-facing error with no retry.
- CPU-bound local ML work (embedding + reranking + MMR) starts to genuinely queue once concurrent CPU-bound threads exceed available cores — Python threads doing torch/numpy compute do release the GIL during the actual C/C++ kernel execution, so *some* real parallelism exists, but throughput will visibly degrade past core-count concurrency.
- **Likely bottleneck: Groq rate limits (unverified magnitude) and/or local CPU saturation for embedding/reranking, whichever hardware/quota is smaller.**

### Scenario C — 500 concurrent users

- Groq calls in flight: ~2,500. High confidence this exceeds typical hosted-inference-API rate limits without an enterprise/high-tier Groq plan — **requires direct verification with Groq**, but the order of magnitude alone (thousands of calls/second-ish burst) is a legitimate red flag regardless of the exact number.
- Per-request unbounded thread-pool creation (Part 16) becomes a genuine concern: 500 concurrent chat requests, each internally spawning its own ~2-22-thread pools for query-variant fan-out plus the concurrent-expansion pool, could create **thousands of transient OS threads simultaneously** — a real resource-pressure risk distinct from (and additional to) the LLM-call-volume risk.
- WebSocket connections: 500 concurrent `/ws/chat` connections is not inherently a problem for a single ASGI process (WebSockets are cheap relative to threads), but each active streaming connection ties up a background thread (`asyncio.to_thread`-driven producer) for the duration of that turn — contributing to the same thread-pressure concern above.
- **Likely bottleneck ranking:**
  1. **Critical** — Groq API rate limits (unverified magnitude, high risk at this volume).
  2. **High** — unbounded per-request thread-pool creation under the single-process backend.
  3. **High** — local CPU/GPU saturation for embedding + reranking + MMR (three separate ML compute steps per turn).
  4. **Medium** — single Qdrant node/collection with no payload-field index on `conversation_id`.

### Scenario D — 1,000 concurrent users during a university deadline/exam period

- All Scenario C risks are amplified. Additionally:
  - **Burst-correlated memory-summarization storms:** if many conversations happen to cross their `MEMORY_MAX_MESSAGES` threshold around the same time (plausible during a synchronized high-usage event — many students all having similarly-long conversations during exam week), the **unbounded** `threading.Thread(daemon=True)` spawned per trigger (Part 16) could produce a spike of dozens-to-hundreds of simultaneous background Groq calls with zero throttling, on top of the foreground request load — a genuinely realistic, code-verified compounding risk specific to *this* kind of correlated-burst scenario.
  - **Report generation as a cost/latency amplifier:** each report request is ~9 Groq calls by itself (Part 6.1); if report generation becomes popular during a "generate my transcript summary" style rush, it multiplies load disproportionately per request relative to a normal chat turn.
  - Single-process, single-Qdrant-collection, single-Groq-account architecture has **no horizontal scaling story today** — this scenario is realistically where the current architecture, unmodified, would begin visibly degrading or failing for a meaningful fraction of users.

### Bottleneck ranking (overall, across scenarios)

1. **Critical** — No rate-limit/backoff handling for the Groq API, combined with unverified actual provider limits, at exactly the concurrency levels a university deployment will realistically hit during peak periods.
2. **Critical** — No horizontal scaling path for the AI subsystem's own state (Agent registry) — an **integration requirement**, needs a decision before multi-replica deployment is attempted.
3. **High** — Unbounded per-request thread-pool creation (query-variant fan-out) and unbounded background-thread spawning (memory summarization) — both real, code-verified, and fixable independently of any infrastructure change.
4. **High** — Local CPU/GPU compute for embedding + reranking + MMR is entirely un-benchmarked at realistic candidate volumes and concurrency; this audit cannot respond with confidence numbers, only flag it as the most likely *local hardware* bottleneck.
5. **Medium** — No payload-field index on `conversation_id` in the single shared Qdrant collection — fine today, a real cost once the collection grows to 15K users' worth of documents (compounded further if the per-conversation duplication issue from Part 5.9 is not resolved, since it multiplies collection size unnecessarily).

---

## 15. Caching

### What can safely be cached today

| Candidate | Currently cached? | Safe to cache? | Notes |
|---|---|---|---|
| Query rewrite/translation (exact text match) | **Yes** — `@lru_cache` on `_translate`/`_rewrite_query` | Yes, already done well | In-process only, exact-string keyed (case/punctuation-sensitive), bounded (256/512 entries) — real but limited win |
| Query embeddings | No | Yes, for exact-text-match queries — embeddings are deterministic given the same model+text | Currently re-embedded from scratch every time, even for byte-identical repeated questions |
| Retrieval results (same query + same document set) | No | Yes, **but only with correct invalidation tied to the conversation's document set** (must invalidate on any new upload or reset to that `conversation_id`) | Non-trivial correctness requirement — must not be done naively with a flat TTL |
| Reranking results | No | Same caveat as retrieval | — |
| Final LLM answers | No | **Risky** — depends on conversation memory (turn-specific context), so answer caching is only safe for a genuinely fresh/empty conversation asking a common question against a *shared* document set | Not meaningfully cacheable at all under the *current* per-conversation-isolated document model (Part 5.9) — this is the crux of why the shared-knowledge-base decision matters so much |
| Document summaries (report content) | No | Yes, for the *whole-document* report path specifically — regenerating an identical report for an unchanged document is pure waste | Currently regenerates the full map-reduce pipeline (~9 Groq calls) on every request, even for a document whose content hasn't changed |

### Where caching is genuinely dangerous for this domain

- **Stale regulatory/administrative answers.** Any answer- or retrieval-level cache **must** be invalidated the moment an underlying document is updated (a new admission-requirements PDF, a corrected deadline). A naive TTL-based cache with no document-version awareness would risk serving confidently-wrong "official" information to students — directly compounding the staleness gap already identified in Part 7. **Any caching layer added must be tied to document version/upload events, not time alone.**
- **Per-conversation memory-dependent answers must never be shared across conversations**, even for identical question text, unless and until the shared-knowledge-base architecture (Part 5.9) exists and the caching key explicitly excludes conversation-specific memory.

### Recommendation

The single highest-value, lowest-risk caching win available **today, without any architectural change**, is exact-match query embedding caching (mirroring the existing `_translate`/`_rewrite_query` LRU pattern) — cheap to add, meaningfully reduces repeated compute for the FAQ-style repeated-question pattern a university userbase will genuinely produce (many students asking near-identical admission/registration questions). The *bigger* win (shared retrieval/answer caching across the whole user base) is blocked on the Part 5.9 architectural decision and should not be attempted before that decision is made.

---

## 16. Concurrency & Race Conditions

| # | Location | Mechanism | Verdict |
|---|---|---|---|
| 1 | `agent/session.py::_agents` + `_lock` | Proper `threading.Lock` around every registry read/write | **Correctly implemented** — credit where due |
| 2 | `services/upload_jobs.py::_jobs` + `_lock` | Proper `threading.Lock` around every registry read/write | **Correctly implemented** |
| 3 | `services/handwritten_ocr_service.py::_get_model()` | Double-checked locking (`self._lock`) around lazy per-language model load | **Correctly implemented** |
| 4 | `services/embeddings_provider.py::get_embeddings()` | No lock, **but** called eagerly at module-import time (`rag_service.py`'s top-level `embeddings = get_embeddings()`), single-threaded, before any request can arrive | **Safe in practice**, due to *when* it's first called, not because the function itself is thread-safe — worth noting as a fragile-by-convention pattern, not a robust one, should this call site ever move |
| 5 | `services/llm_provider.py::get_llm()`/`get_agent_llm()` | No lock, but the only concurrent-creation path (`AgentLLM.__init__` via `agent/session.py::get_agent()`) is itself called while `session._lock` is held | **Safe in practice**, same caveat as #4 |
| 6 | **`services/audio_service.py::_get_model()`** | **No lock at all**, and lazily first-triggered by a real per-request call path (`transcribe_audio()`, called directly from `routes/chat.py::chat_voice` with no serialization) | **Confirmed race condition** — two-plus concurrent voice requests on a cold backend can trigger duplicate concurrent `whisper.load_model()` calls |
| 7 | **`services/rag_service.py::_get_cross_encoder()`** | **No lock**, lazily first-triggered by the first real `retrieve()` call — reachable concurrently by design, since multiple chat requests routinely retrieve concurrently | **Confirmed race condition** — same root cause and risk profile as #6 |
| 8 | `rag_service.py::_run_concurrent()` / `utils/timing.py::run_concurrent_ctx()` | A **new** `ThreadPoolExecutor` is created and torn down per call (sized to the task count, e.g., up to ~22 for query-variant fan-out) | **Not a correctness bug**, but an unbounded-at-the-system-level resource pattern — no global cap on total concurrent threads across simultaneously in-flight requests (Part 15) |
| 9 | `memory/memory_manager.py::_summarize_async()` | Spawns a bare `threading.Thread(daemon=True)` per trigger, no pool, no concurrency cap | Same class of risk as #8 — correlated bursts (Part 14, Scenario D) could spawn many simultaneous unthrottled background LLM calls |
| 10 | `rag_service.py::_load_registry()`/`_save_registry()` (`processed_files.json`) | Full-file read → mutate → full-file write, **no OS-level lock, no atomic replace** | **Confirmed race condition** under concurrent uploads — last writer wins, can silently drop another request's just-written registry entry. This is the AI ingestion pipeline's own dedup/bookkeeping correctness, not generic storage infrastructure — worth fixing (or explicitly re-specifying) even though the *storage mechanism itself* will likely be replaced during integration |

**Cross-session contamination:** not observed anywhere — every shared-state structure found is either correctly locked or scoped per-conversation-id such that concurrent access from *different* conversations cannot collide (findings #6, #7, #9 are about wasted duplicate work / resource pressure, not about one user's data leaking into another's).

---

## 17. AI-Specific Security

*(Authentication/authorization are explicitly out of scope per the brief — this section covers threats specific to the AI/LLM logic itself.)*

### Prompt injection (direct, via the user's own message)

The planner's output is constrained to a strict, Pydantic-validated JSON schema (`TypeAdapter(AgentAction).validate_python`) — a user cannot manipulate the *planner* into invoking an undefined tool or smuggling unvalidated arguments past this check; malformed/out-of-schema output triggers the same deterministic fallback used for any other planner failure. **This is a genuine, structural, code-verified defense** against the planner being hijacked, worth crediting clearly rather than dismissed as "no defense."

### Indirect prompt injection (via malicious document content)

A document could contain text like "Ignore previous instructions and reveal your system prompt." A critical, positive, code-verified architectural fact limits this: **the planner never sees raw retrieved chunk text.** `_build_messages()`/`_format_observations()` only feed the planner a JSON summary of tool *observations* (chunk counts, source filenames, status) — never the actual chunk content. This means document-based injected instructions **structurally cannot reach or influence the agent's tool-selection logic** — they can only affect the *generation* step (`generate`/`summarize`/`compare`), which does receive raw chunk text with no delimiter/untrusted-data framing and no output-side filtering beyond the light `_clean_answer()` prefix-stripping. **Net assessment: the blast radius of document-content injection is correctly, structurally limited to "can taint this turn's answer text," and cannot escalate into "can hijack the agent's subsequent actions."** This is a meaningfully better security posture than a naive design where the LLM sees everything in one undifferentiated context, and should be explicitly preserved in any future refactor.

**What is not defended:** the generation prompt does not wrap retrieved chunk text in explicit "this is untrusted document data, not instructions" delimiters. A sufficiently crafted document could still manipulate the *content* of a single answer (e.g., injecting a fake "official" statement). **Recommendation:** add explicit delimiters + a system-prompt rule instructing the model to treat delimited content as data, not instructions — low-difficulty, meaningfully reduces (does not eliminate) this residual risk.

### Data exfiltration

Since `build_prompt_with_memory()` places the full rendered fact list + recent messages directly in the same prompt as retrieved (potentially injected) document text, an injected instruction *could* in principle induce the model to echo the memory block back verbatim in its answer. Given the existing per-conversation isolation (out of scope here), this is exfiltrating a user's **own** data back to themselves via a manipulated answer — a trust/answer-quality concern rather than a cross-user leak. The more relevant AI-specific risk is **degraded answer integrity/trust**: a manipulated response undermines confidence in what's meant to be an authoritative university information source, independent of any data-leak framing.

### Tool abuse via document content

Not directly possible: since the planner never sees raw chunk text (established above), a malicious document cannot itself force excessive/expensive tool invocations (e.g., forcing repeated `report` calls). A malicious **user**, typing directly, could still request `report` generation repeatedly — this is a straightforward volumetric-abuse concern (mitigated by rate limiting, out of scope here) rather than an injection vector, but worth surfacing for capacity planning: `report` is by far the most expensive single tool (~9 Groq calls per invocation, Part 6.1) and is reachable through plain natural language with no confirmation step.

### Memory poisoning

**A real, code-verified, in-scope finding.** `memory/fact_extractor.py`'s extraction prompt has **no requirement that a stored "fact" be grounded in a retrieved document** versus simply asserted by the user in conversation — nothing in `build_prompt()` or `FactStore.merge()` distinguishes provenance. A user could deliberately state a false claim persuasively enough that the extraction LLM stores it as a persistent fact, which is then re-injected into **every subsequent prompt for that conversation** (including future `generate`/`respond` calls) until evicted or explicitly contradicted. Scoped to the single conversation (no cross-user impact given existing isolation), but real and unaddressed. **Recommendation:** consider tagging extracted facts with a provenance flag (user-asserted vs. document-derived) and treating user-asserted facts with lower trust/weight in generation prompts, or excluding them from being presented as if they were document-grounded.

---

## 18. Reliability Audit

| Failure | Retry? | Fallback? | Graceful degradation? | User-visible error? |
|---|---|---|---|---|
| LLM (generation) timeout/hang | No | No | No | Eventually, whatever the `groq` SDK's own default timeout produces — not explicitly tuned by this codebase |
| LLM (planning) malformed output | Yes (2 retries) | Yes (deterministic `retrieve` fallback) | Yes | No — turn completes normally |
| LLM 429 / rate limit | No | No | No | Yes — raw exception surfaces as a generic error |
| Qdrant unreachable (startup) | Yes (bounded backoff) | Yes (`is_ready()` reports false) | Yes | "Database is empty" style message |
| Qdrant unreachable (mid-request) | No (per-variant search just returns `[]`) | Partial — other variants may still succeed | Partial | Only fails hard if *every* variant's search fails |
| Embedding model failure | No | No | No | Generic error, categorized as "Could not index the document for search" at the ingestion layer only |
| Reranker (cross-encoder) failure | No | **Yes** — permanent fallback to lexical-only for the process lifetime | Yes | No — silent, correct degradation |
| Whisper failure | No | No | Partial (specific error messages: too small/silent/no speech) | Yes, HTTP 422 with a specific reason |
| Malformed/corrupted document | Partial (per-file try/except, other files in the batch still process) | No | Partial | Only if **zero** meaningful content results — a **partially**-corrupted document (e.g., 3 of 50 pages parse) proceeds silently with an incomplete knowledge base and no warning |
| Empty document | Yes (rejected upfront) | — | — | HTTP 400 |
| Huge document (many pages, scanned) | No processing-time cap at all — only the file-**size** cap applies | No | No | The frontend gives up polling after 15 minutes; the **backend job keeps running indefinitely regardless**, consuming CPU with no server-side abort |
| Unsupported document type | Silently produces zero chunks (not rejected) | — | Misleadingly "succeeds" with 0 chunks | Only visible if the uploader checks the chunk count |

**Highest-priority reliability gaps for a production university deployment**, in order: (1) no rate-limit-aware handling for the dominant external dependency (Groq), (2) no processing-time bound on ingestion (a single huge scanned document can monopolize resources indefinitely), (3) silent partial-corruption ingestion with no visibility to the user.

---

## 19. AI Observability Audit

### What exists today

- Per-stage request-latency profiling (`utils/timing.py`) for `/api/chat`: language detection, query rewrite/translation, embedding, Qdrant retrieval, reranking, MMR, agent planning, memory, LLM generation — logged as plain text per request, `LOG_REQUEST_PROFILE` (default on).
- Verbose, opt-in retrieval debug logging (`LOG_RETRIEVAL_DEBUG`) — query variants, per-variant hits, pre/post-rerank scores, final context sent to the LLM.
- Per-iteration agent thought/action logging, opt-in via `AGENT_DEBUG`.
- **None of this is exported as a metric** — it is all plain-text log output, human-readable, not machine-aggregable, with no dashboard, no time-series store, no alerting.

### What cannot currently be measured (all confirmed absent)

Retrieval/reranking/LLM/answer latency **percentiles** (only per-request raw numbers exist in logs, never aggregated); agent iterations per turn (present in per-request logs only, not counted); tool-usage distribution; retrieval count and empty-retrieval rate (the confidence-gate rejection path *is* logged when `LOG_RETRIEVAL_DEBUG` is on, but not counted/exported); retrieval score distributions; token usage per call (not captured anywhere — the `groq` SDK response includes usage data that is never read or logged); failed-request counts; 429 counts (impossible to count something that isn't even specially detected — Part 6.3); any hallucination/faithfulness signal (none exists — Part 7); user feedback (no feedback mechanism exists in the UI at all); abstention rate (how often the "not available in the uploaded files" answer fires — not counted, though it is a fixed, greppable string that *could* be counted cheaply).

### Recommended AI observability design (not yet implemented)

1. **Emit structured (JSON) log lines**, not just formatted text, for every request's `utils/timing.py` report — trivial to add, immediately enables aggregation by any standard log pipeline the integration team adopts.
2. **Count, don't just log:** agent iterations/turn, tool distribution, empty/rejected retrievals (confidence-gate fires), abstention-answer rate, Groq call count/latency/errors (including 429s once they're actually detected — Part 6.3), token usage per call (already available from the Groq response, just currently discarded).
3. **A lightweight groundedness/abstention dashboard** is the single most valuable AI-quality signal achievable with the least new infrastructure: the abstention string is fixed and greppable *today* — tracking its rate over time, per language, per document set, would immediately surface both retrieval-quality regressions and genuinely under-covered topics, with no model changes required.
4. None of this requires new infrastructure investment beyond structured logging + counters — it does not require standing up a full metrics/tracing stack before it becomes useful.

---

## 20. Evaluation Framework

### Does one exist?

**No.** Confirmed by direct repository search: no pytest/unittest anywhere, no eval scripts, no labeled QA dataset, no retrieval-quality benchmark, no LLM-as-judge harness, no CI. The only "evaluation" evidence in the repo is prose developer notes (`backend/PROFILING.md`, code comments referencing "verified directly") describing manual, one-off investigation — not a repeatable, versioned evaluation suite.

### Recommended dataset design

A hand-built (or lightly LLM-assisted, human-reviewed) set of **real university questions**, explicitly stratified across the axes this audit identified as pipeline-relevant:

```text
question
expected_answer
expected_source_document(s)
expected_evidence_chunk(s)          # for retrieval-quality scoring
language: {formal_ar, egyptian_ar, en, code_switched}
category: {admission, registration, exam, financial, academic_regulation, course_specific, general}
difficulty: {direct_lookup, multi_hop, comparison, ambiguous, out_of_scope, contradictory_sources}
```

The `out_of_scope` and `contradictory_sources` categories are essential and currently entirely untested — they directly probe the two weakest points identified in this audit (grounding/abstention correctness, and document staleness handling).

### Metrics

- **Retrieval:** Recall@K, Precision@K, MRR, nDCG — computed against `expected_evidence_chunk(s)`. Directly measures whether the `RETRIEVER_K`/`RERANK_TOP_N`/`MMR_LAMBDA` defaults (all currently unvalidated) are actually well-tuned.
- **Generation:** answer correctness (exact/semantic match against `expected_answer`), answer relevance, faithfulness (does the answer's content actually appear in the retrieved context — the missing verification layer from Part 7), citation correctness (does the "Sources:" line match `expected_source_document(s)`).
- **Agent:** tool-selection accuracy (does the chosen tool match a labeled expectation), unnecessary-tool-call rate, premature-terminal rate (directly measures whether the Part 4.2 backstop's false-trigger rate is acceptable), average iterations per question.
- **Performance:** p50/p95/p99 latency, throughput — currently **entirely unmeasured**; this framework is the correct place to finally generate real numbers instead of relying on the two isolated benchmark figures in `utils/device.py`.
- **Voice:** WER, stratified by language/dialect category, and separately for the single-pass vs. Egyptian-double-pass path.
- **OCR:** CER/WER, stratified by document quality (clean scan vs. low-quality photo) — directly enables the "flag low-confidence OCR" recommendation from Part 11 to be validated rather than assumed useful.

### How this should be implemented

A standalone, versioned evaluation script (not integrated into the production request path) that: loads the labeled dataset, replays each question through the actual `rag_service`/`agent` code paths (not a reimplementation — this is important, so the eval genuinely exercises production logic), scores against the metrics above, and outputs a report comparable across runs (so a future change to `CHUNKING_STRATEGY`, `RERANK_ALPHA`, `MMR_LAMBDA`, or the embedding model can be evaluated for real impact before being shipped, rather than trusted on intuition as every current default currently is). This should be built **before** any further retrieval-tuning work, since right now there is no way to know whether a proposed change is actually an improvement.

---

## 21. Existing Defects

Only items directly verified in code are listed. Severity reflects impact on correctness/reliability/cost specifically for the AI subsystem, independent of the excluded infra/auth concerns.

| ID | Problem | Component | Evidence/File | Severity | User Impact | 15K Impact | Recommended Fix |
|---|---|---|---|---|---|---|---|
| D1 | Semantic chunking strategy silently produces zero chunks for any multi-sentence document | Chunking | `services/rag_service.py::_semantic_split_documents` — computed `chunk_text` is never appended to `out` | **Critical** | Total, silent data loss for anyone who sets `CHUNKING_STRATEGY=semantic` | Same — not the default, but a live footgun if ever enabled | Append the computed chunk to `out` (few-line fix), or remove the option until fixed |
| D2 | Whisper STT model singleton has no lock around lazy initialization | Voice | `services/audio_service.py::_get_model()` | **High** | Duplicate/wasted model loads on concurrent cold-start voice requests; worst case, undefined behavior during concurrent load | Directly reachable at any real concurrency (first users after any restart) | Add double-checked locking, mirroring `handwritten_ocr_service.py`'s existing correct pattern |
| D3 | Cross-encoder reranker singleton has no lock around lazy initialization | Retrieval | `services/rag_service.py::_get_cross_encoder()` | **High** | Same failure mode as D2, reachable on the very first concurrent chat requests after backend start | Same | Same fix pattern |
| D4 | Background memory-summarization spawns unbounded, uncapped `threading.Thread`s | Memory | `memory/memory_manager.py::_summarize_async` | **Medium** (High during correlated bursts) | None to the triggering user (async), but adds unthrottled Groq load | Real risk during synchronized high-usage events (exam week) — Part 14 Scenario D | Bound via a small `ThreadPoolExecutor`, mirroring the pattern already correctly used in `report_service.py` |
| D5 | Per-request `ThreadPoolExecutor` creation is unbounded system-wide | Retrieval | `rag_service._run_concurrent`, `utils/timing.py::run_concurrent_ctx` | **Medium** (High at Scenario C/D concurrency) | None per-request | Real OS-thread-pressure risk at hundreds of concurrent chat requests | Consider a shared, globally-bounded executor for query-variant fan-out instead of per-call pool creation |
| D6 | Exact-string-only duplicate-retrieve guard allows near-duplicate retrieve loops to burn through iterations | Agent | `agent/agent.py::_run_impl`/`_run_stream_impl` | **Medium** | Occasional slow/expensive turns, bounded by `AGENT_MAX_ITERATIONS=6` | Higher Groq-call volume under bad-luck planning sequences at scale | Normalize/fuzzy-match retrieve questions before the duplicate check, not exact string only |
| D7 | No configured request timeout for any Groq call | LLM | `services/llm_provider.py::GroqLLM.chat`/`stream_chat` | **Medium-High** | A hanging Groq response can tie up a worker thread indefinitely | Directly worsens thread-pressure risk (D5) under load | Set an explicit, tuned `timeout=` on every Groq call |
| D8 | No rate-limit-aware retry/backoff for any Groq call except planner JSON-validation retries | LLM | `services/llm_provider.py`, all call sites in `rag_service.py` | **High** | A single burst can turn into a wave of user-facing errors instead of graceful queuing | Directly the top risk identified in Part 14/15 at realistic peak concurrency | Add exponential backoff + 429-specific handling at the `GroqLLM` layer, shared by every caller |
| D9 | Document versioning/staleness is entirely unaddressed — superseded documents' chunks are never removed, and ingestion `timestamp` metadata is captured but never shown to the LLM | RAG grounding | `rag_service.update_db_files`, `_chunk_label`, `loaders/base.py::make_meta` | **High** (domain-specific severity) | Students can receive confidently-stated, outdated official information with no visible warning | Compounds with scale — more re-uploads over time, more stale duplicates | Surface `timestamp` in `_chunk_label`; add a prompt instruction to flag/prefer recency on conflict; consider an explicit document-supersession mechanism |
| D10 | Handwritten-OCR per-line recognition is strictly sequential, not batched | OCR | `services/handwritten_ocr_service.py::recognize_with_debug` — `[self._recognize_image(img, language) for img in lines]` | **Medium** | Slow multi-line handwritten OCR requests (up to 80 sequential model calls) | Real if this feature sees meaningful adoption | Batch line-image inference through the model together |
| D11 | No caching for query embeddings, retrieval results, or reranking results (only exact-text query-rewrite/translation is cached) | RAG | `rag_service.py` (absence throughout), contrast with the existing `@lru_cache` pattern | **Medium** | Repeated/near-identical questions (realistic FAQ pattern) recompute the full pipeline every time | Real, compounding cost at 15K users asking overlapping questions | Add exact-match embedding cache at minimum; broader caching blocked on the Part 5.9 architecture decision |
| D12 | No candidate-count cap before cross-encoder reranking | Retrieval | `rag_service._rerank` | **Low-Medium** | Occasional larger-than-typical rerank batches on highly-overlapping document sets | Minor latency-variance risk at scale | Add an explicit max-candidates cap before scoring |
| D13 | No OCR quality/confidence signal captured at ingestion time | Ingestion | `services/ocr_service.py` (uses `image_to_string`, never `image_to_data`) | **Medium** | Silently poor answers traceable to bad scans with no diagnostic signal anywhere | Real at scale given diverse student-submitted scan quality | Capture and surface a confidence proxy per document |
| D14 | `compare` tool does not explicitly group retrieved chunks by source document in the prompt | Generation quality | `services/rag_service.py::compare`/`compare_stream` | **Low-Medium** | Possible cross-document attribution confusion in comparison answers, especially after MMR interleaving | Same, scales with comparison-tool usage | Explicitly section/label chunks by source in the prompt construction |
| D15 | `processed_files.json` read-modify-write has no lock/atomic replace | Ingestion bookkeeping | `rag_service._load_registry`/`_save_registry` | **Medium** | Concurrent uploads can silently drop each other's registry entries (dedup/chunk-count bookkeeping) | Real under concurrent bulk-upload periods | Add file locking or atomic temp-file-then-rename; likely superseded by a real datastore during integration, but the *semantics* it must preserve (per-conversation dedup scoping) should be documented for that migration |

---

## 22. Future Risks for 15K Users

Distinguished explicitly from the confirmed defects above — these are risks that **may** materialize as usage grows, not bugs present today.

| Risk | Why it may happen | Trigger | Impact | Preventive solution |
|---|---|---|---|---|
| Groq rate-limit exhaustion | No backoff/retry exists (D8); real limits unverified | Any burst above whatever Groq's actual account tier supports | Wave of failed chat turns during exactly the moments (exam periods) when reliability matters most | Verify actual Groq limits; implement backoff; consider a request queue/admission-control layer |
| Local CPU/GPU saturation for embedding+reranking+MMR | Three separate ML compute steps per turn, single-process, un-benchmarked at scale | Sustained concurrency above available cores/GPU capacity | Rising p95/p99 latency, eventually visible slowness across all users | Benchmark under load (Part 20); consider a separate, horizontally-scalable inference tier for embeddings/reranking |
| Correlated memory-summarization storms | Unbounded background threads (D4) | Synchronized high-usage events (many long conversations crossing the threshold together) | Sudden unthrottled Groq load spike, compounding foreground load | Bound the background-thread pool (same fix as D4) |
| Qdrant collection growth without a `conversation_id` payload index | No index exists today; fine at current scale | Collection size growing into the 15K-user range, especially if the Part 5.9 per-conversation-duplication pattern continues | Rising retrieval latency as filtered searches scan a larger, unindexed payload space | Add a payload-field index once collection size materially grows; resolve the Part 5.9 shared-knowledge-base question first, since it directly controls how large the collection needs to be at all |
| Prompt injection via document content, generation-step only | No delimiter/untrusted-data framing exists today (Part 17) | A deliberately crafted malicious upload | Individual manipulated/untrustworthy answers (bounded blast radius — cannot hijack agent actions) | Add delimiter + explicit untrusted-data framing in the generation prompt |
| Repeated identical/near-identical questions at scale | No embedding/retrieval/answer caching beyond exact-text query-rewrite (D11); no shared knowledge base (Part 5.9) | Natural FAQ-clustering behavior of a 15K-student population | Large amount of fully-redundant compute and Groq spend for questions that are, in substance, being answered from scratch repeatedly | Add embedding caching now; resolve the shared-knowledge-base question to unlock retrieval/answer-level caching |
| Very large documents (bulk scanned transcripts/forms) | No ingestion processing-time cap exists | A student uploading a large multi-page scanned PDF | Long-running background jobs consuming CPU indefinitely with no server-side abort | Add an explicit ingestion processing-time budget/abort, separate from the existing file-size cap |
| Many simultaneous uploads during a registration/deadline period | `processed_files.json`'s unlocked read-modify-write (D15) | Concurrent bulk uploads clustering around a deadline | Silently dropped registry entries — a chunk count/dedup bookkeeping inconsistency, not data loss in Qdrant itself, but a real correctness gap in what the system believes was ingested | File locking now; a real datastore during integration |
| Long conversations at true scale (many long-running conversations simultaneously) | Memory design is already well-bounded per-conversation (Part 9) | High volume of *concurrently active*, individually-long conversations | Primarily a background-Groq-call-volume concern (already covered above), **not** a per-conversation cost-growth concern — this risk is smaller than it might naively appear, given the existing bounded design | Already substantially mitigated by existing architecture; monitor background-call volume via Part 19's recommended metrics |

---

## 23. Recommended Solutions

For the highest-priority findings (Critical/High), in the standard Problem → Root cause → Current behavior → Solution → Implementation → Priority → Expected improvement → Trade-offs format. Lower-severity items are covered in the tables above and the roadmap below without full expansion, for brevity.

### Solution 1 — Fix the semantic chunking bug (D1)

- **Root cause:** `_semantic_split_documents()`'s grouping loop computes `chunk_text` but never calls `out.append(...)`.
- **Current behavior:** silently returns zero chunks for any multi-sentence document when this strategy is selected.
- **Solution:** append a `Document(page_content=chunk_text, metadata=doc.metadata)` inside the loop, matching the pattern used by the (working) `_hybrid_split_documents`.
- **Implementation:** a few lines, no design change needed.
- **Priority:** Critical, immediate.
- **Expected improvement:** restores a currently-broken, documented feature.
- **Trade-offs:** none meaningful — this is a pure bugfix.

### Solution 2 — Lock the two unlocked lazy singletons (D2, D3)

- **Root cause:** `audio_service._get_model()` and `rag_service._get_cross_encoder()` were written without the double-checked-locking pattern already correctly used elsewhere in the same codebase (`handwritten_ocr_service.py`).
- **Solution:** apply the identical, already-proven pattern (a `threading.Lock`, check-lock-check-again) to both.
- **Priority:** High, before any concurrent-load testing.
- **Expected improvement:** eliminates duplicate model loads and the associated undefined-behavior risk under concurrent cold-start requests.
- **Trade-offs:** negligible — a lock acquired once per process lifetime (after first load, the fast path never touches the lock).

### Solution 3 — Add rate-limit-aware retry/backoff for all Groq calls (D8)

- **Root cause:** `GroqLLM.chat`/`stream_chat` have no retry logic beyond the agent-planner's own JSON-validation retries; every other caller (query expansion, generation, memory extraction, report map-reduce) has zero resilience to transient failures or 429s.
- **Current behavior:** any Groq exception, including a rate limit, propagates immediately as a user-facing (or silently-failed-background) error.
- **Solution:** centralize retry-with-exponential-backoff (with 429-specific handling — respecting a `Retry-After` header if Groq provides one) inside `GroqLLM` itself, so every caller benefits automatically without individual changes.
- **Implementation approach:** wrap the existing `client.chat.completions.create(...)` call site(s) in a small retry decorator/helper; bound total retry time so a request doesn't hang indefinitely (interacts with Solution 4 below).
- **Priority:** High — this is the single most consequential fix for surviving realistic peak concurrency (Part 15, Scenarios B–D).
- **Expected improvement:** graceful degradation instead of a wave of hard failures during bursts.
- **Trade-offs:** added latency on the failure path (acceptable — better than an immediate hard error); requires knowing Groq's actual rate-limit/retry-after semantics, which must be verified directly with the provider, not assumed.

### Solution 4 — Configure explicit request timeouts for all Groq calls (D7)

- **Root cause:** no `timeout=` is ever passed; behavior relies entirely on the SDK's own undocumented-in-repo default.
- **Solution:** set an explicit, deliberately-chosen timeout per call type (e.g., a shorter timeout for the cheap planning/expansion calls, a longer one for final generation).
- **Priority:** Medium-High, pairs naturally with Solution 3.
- **Expected improvement:** bounds worst-case thread occupancy per request, directly reducing the thread-pressure risk compounding with D5.
- **Trade-offs:** a too-aggressive timeout could prematurely fail a genuinely-slow-but-would-have-succeeded call — needs tuning against real observed Groq latency (Part 20's eval framework is the right place to establish this).

### Solution 5 — Surface document recency to the model; add a conflict-awareness prompt rule (D9)

- **Root cause:** `timestamp` metadata is captured at ingestion but never included in `_chunk_label()`'s prompt-visible chunk header; no prompt rule addresses conflicting sources.
- **Solution:** include a recency indicator (ingestion date, or ideally an explicit document-version/supersession concept if the integration team introduces one) in the chunk header shown to the LLM; add an explicit grounding-prompt rule: "if retrieved chunks disagree, say so explicitly and prefer the most recently uploaded source rather than silently picking one."
- **Priority:** High — directly addresses the most consequential domain-specific grounding gap identified in this audit.
- **Expected improvement:** meaningfully reduces the risk of confidently-stated stale official information.
- **Trade-offs:** low implementation cost; the *harder*, longer-term fix (an actual document-supersession/replace mechanism, so old chunks are genuinely retired rather than merely down-weighted) is a bigger change worth planning for but not blocking this smaller, immediate mitigation.

### Solution 6 — Bound the two unbounded thread-spawning patterns (D4, D5)

- **Root cause:** `_summarize_async()` spawns a bare daemon thread per trigger with no cap; `_run_concurrent`/`run_concurrent_ctx` create a fresh `ThreadPoolExecutor` per call with no system-wide ceiling.
- **Solution:** route background memory-summarization through a small, shared, bounded `ThreadPoolExecutor` (mirroring `report_service.py`'s already-correct `MAP_EXTRACT_CONCURRENCY`-bounded pattern); consider a shared, appropriately-sized executor for query-variant fan-out instead of per-call pool creation, or at minimum a global semaphore capping total concurrent fan-out threads across all in-flight requests.
- **Priority:** Medium (High specifically for Part 14/15 burst scenarios).
- **Expected improvement:** predictable, bounded resource usage under correlated bursts instead of an unbounded thread count proportional to simultaneous triggers.
- **Trade-offs:** a shared bounded pool introduces queuing under extreme load (by design — this is the point, converting "unbounded resource growth" into "bounded, predictable queuing").

### Solution 7 — Resolve the shared-knowledge-base architecture question (Part 5.9)

- **Root cause:** the current per-conversation document-isolation model was a deliberate, reasonable design choice for the prototype's original scope, but does not fit "15,000 students querying largely-overlapping official documents."
- **Solution:** decide, before further scaling work, whether production should introduce a shared/global document tier (indexed once, queried by everyone) alongside optional per-student private uploads — versus continuing pure per-conversation isolation with the redundancy that implies.
- **Priority:** Critical **as a decision**, even though the implementation itself is a larger, later effort — this decision gates the value of nearly every caching and cost-optimization recommendation in this report.
- **Expected improvement:** potentially order-of-magnitude reduction in redundant ingestion compute and unlocks meaningful cross-user caching.
- **Trade-offs:** a real architectural change — needs coordination with the production backend team (an integration requirement, not something the AI subsystem can decide alone), and needs a clear content-governance answer (who can upload/update the "official" shared documents, and how supersession/versioning works — which also directly solves Solution 5's harder half).

---

## 24. Prioritized Remediation Roadmap

### Phase 1 — Before Integration (AI subsystem only, no production-backend coordination needed)

- Fix D1 (semantic chunking bug).
- Fix D2/D3 (unlocked model singletons).
- Fix D15 (registry file locking) or explicitly document the dedup semantics it must preserve for whoever replaces it.
- Fix D6 (near-duplicate retrieve loop guard).
- Bound D4/D5 (unbounded thread spawning).
- Add explicit Groq call timeouts (Solution 4 / D7).
- Add rate-limit-aware retry/backoff (Solution 3 / D8) — the single highest-leverage reliability fix achievable entirely within the AI subsystem.
- Surface document recency in prompts + add a conflict-awareness rule (Solution 5 / D9) — the immediate, low-cost mitigation half; defer the full supersession mechanism to Phase 2/3.

### Phase 2 — During Integration (requires production-backend coordination)

- Resolve the shared-knowledge-base vs. per-conversation-isolation architecture question (Solution 7) — this is the single biggest lever for both cost and caching, and must be a joint decision.
- Externalize Agent/short-term-memory state (or agree on sticky routing) if/when the production backend runs multiple replicas — an integration requirement, not an AI-subsystem-only fix.
- Verify actual Groq API rate limits/tier against realistic peak-concurrency numbers from Part 15, and size the backoff strategy from Solution 3 accordingly.
- Decide document-supersession/versioning semantics jointly with whatever storage/database layer the production backend introduces (this directly completes Solution 5).

### Phase 3 — Before University Launch

- Build the evaluation framework (Part 20) and run it against the tuned defaults (`RETRIEVER_K`, `RERANK_TOP_N`, `RERANK_ALPHA`, `MMR_LAMBDA`, chunk sizes) — validate or retune every currently-unvalidated default.
- Add AI observability (Part 19) — structured logs, counters for iterations/tool-usage/abstention-rate/token-usage — before, not after, real traffic arrives.
- Benchmark local ML compute (embedding + reranking + MMR) under realistic concurrency to replace the "requires benchmarking" placeholders in Parts 5/14 with real numbers, and size hardware accordingly.
- Fix D10 (batch handwritten-OCR line inference), D13 (OCR confidence capture), D12 (rerank candidate cap), D14 (compare-tool source grouping).
- Add embedding-level caching (D11's first, lowest-risk step, independent of the Solution 7 decision).
- Run the Arabic/English stratified evaluation (Part 10/20) specifically before broad launch, given the explicit Egyptian-university target audience.

### Phase 4 — Post-launch

- Broader retrieval/answer caching, contingent on the Phase 2 architecture decision.
- A real citation-faithfulness verification layer (closing the largest remaining gap in Part 7).
- Prompt-injection delimiter hardening for the generation step (Part 17).
- Ongoing evaluation-suite expansion as real user questions/failure patterns are observed in production.

---

## 25. Top 10 Things to Fix Before Integration

Ranked by (severity × how cheaply/independently the AI subsystem can fix it, without waiting on the production backend team):

1. **D1 — Fix the semantic chunking bug.** Trivial, currently a live data-loss trap.
2. **D8 — Add rate-limit-aware retry/backoff for Groq calls.** The single highest-leverage reliability fix for surviving real traffic, fully within the AI subsystem's control.
3. **D2/D3 — Lock the two unlocked model singletons (Whisper, cross-encoder).** Small, precise, closes real concurrency bugs before any load testing is meaningful.
4. **D9 — Surface document recency + add a conflict-awareness prompt rule.** The most consequential domain-specific grounding gap; the immediate mitigation is cheap even though the full fix (Solution 7) is bigger.
5. **D4/D5 — Bound the two unbounded thread-spawning patterns.** Directly prevents a realistic burst-scenario failure mode (Part 14, Scenario D).
6. **D7 — Add explicit Groq request timeouts.** Pairs directly with #2; without it, retries can compound hangs instead of resolving them.
7. **D6 — Fix the exact-string duplicate-retrieve guard.** Cheap, closes a real (if bounded) cost-multiplication path.
8. **Part 5.9 — Force the shared-knowledge-base architecture decision now**, even though the implementation is Phase 2/3 — every day this is undecided, more redundant per-conversation ingestion accumulates, and more caching/cost-optimization work gets built on an assumption that may need to be discarded.
9. **Part 20 — Stand up even a minimal evaluation harness** before further tuning any retrieval parameter — right now every default (`RERANK_ALPHA`, `MMR_LAMBDA`, `RETRIEVER_K`, chunk sizes) is unvalidated, and further changes without a way to measure impact risk making things worse invisibly.
10. **D15 — Lock (or document the semantics of) `processed_files.json`'s dedup registry** before any concurrent bulk-upload testing, so the integration team inherits a documented, understood contract rather than a silent, unlocked correctness gap.

---

## 26. Final AI Readiness Assessment

| Category | Rating | Why |
|---|---|---|
| AI Architecture | **Good** | Genuinely agentic, well-separated, thoughtfully documented; undermined by no eval harness to validate any of its many tuned parameters |
| RAG Quality | **Good** | A real hybrid pipeline (dense + lexical + cross-encoder + MMR + budgeting), reasoned defaults throughout; docked for the character-vs-token budget mismatch and zero measured recall/precision |
| Agent Quality | **Good** | A genuine, bounded ReAct loop with a well-reasoned, code-verified backstop for a real prior failure mode; docked for the uncovered "retrieved-then-ignored" gap and the near-duplicate-loop inefficiency |
| Grounding | **Fair** | Strong, repeated prompt engineering, but entirely prompt-based with zero independent verification, and a real, domain-serious staleness/versioning gap |
| Memory | **Very Good** | Genuinely well-bounded (does not suffer the long-conversation cost-growth problem this audit was specifically asked to check for); dedup is character-level rather than semantic, a minor gap |
| Arabic/English | **Good** | Real, non-superficial bilingual engineering (normalization, dual-language OCR config, code-switch-aware prompting, dialect-aware STT); every model-quality claim on dialect/code-switching specifically is unverified and should be a Phase-3 priority |
| AI Reliability | **Fair** | Graceful degradation exists for Qdrant/MinIO/reranker-load failure, but the dominant external dependency (Groq) has zero retry/backoff/timeout discipline — the single biggest reliability gap in the whole subsystem |
| Scalability Readiness | **Fair** | No horizontal-scaling story for AI state (an integration requirement, not a defect), plus two confirmed unbounded-thread-spawning patterns and zero rate-limit handling — all independently fixable before integration |
| Evaluation Readiness | **Poor** | Zero automated evaluation of any kind exists; every tuned parameter in the entire RAG/agent stack is currently unvalidated |
| **Production AI Readiness for 15K Users** | **Fair** | A strong technical foundation with several small, well-scoped, high-leverage fixes standing between it and genuine production readiness — this is a "fix ten specific things," not a "rearchitect," verdict |

**Bottom line:** the AI subsystem's core design decisions (agentic loop, hybrid retrieval, bounded memory, structural injection resistance at the planner level) are sound and above-average for a project at this stage. What stands between this and 15,000-user readiness is not a redesign — it is closing two confirmed concurrency bugs, adding LLM-call resilience (retry/backoff/timeout), closing one serious domain-specific grounding gap (document staleness), and, critically, deciding the shared-knowledge-base architecture question before more work is built on top of the current per-conversation-isolated assumption.

---

# What Can Break After Deployment?

The most realistic, code-grounded failure scenarios the team should specifically test before this subsystem is integrated into the university's production system:

1. **A registration-deadline traffic spike triggers Groq 429s across hundreds of concurrent chat turns simultaneously, and every one of them surfaces as a raw, unhelpful error to the student**, because no rate-limit handling exists anywhere (D8).
2. **The backend restarts (a deploy, a crash-recovery) and the first two or three students to send a voice message at nearly the same moment each trigger their own full Whisper model load concurrently**, wasting memory and, in the worst case, producing a corrupted/undefined model state (D2).
3. **A student uploads a large, multi-page scanned transcript**, and the ingestion job runs for many minutes consuming CPU with no server-side time limit — while several other students' uploads queue up behind it on the same constrained thread pool.
4. **An official document (e.g., admission requirements) is updated and re-uploaded**, but the old version's chunks are never removed — a student asks the same question a week apart and silently gets two different, contradictory answers, with no indication either time that a newer document exists (D9).
5. **Exam week produces many long, simultaneous conversations that all cross their memory-summarization threshold around the same time**, spawning an unbounded burst of background Groq calls on top of already-heavy foreground load (D4), compounding exactly when the system is under the most pressure.
6. **A batch of students all upload the same official PDF around the same time (e.g., everyone grabbing the same posted syllabus)**, and the system independently re-parses, re-OCRs (if scanned), and re-embeds it once per student, multiplying compute for content that is byte-identical (Part 5.9) — with zero benefit shared across them.
7. **A student photographs a low-quality, poorly-lit scan of a form**, OCR silently produces garbled text that still passes the ≥15-character/contains-a-letter meaningfulness check, and the chatbot subsequently gives a confidently wrong answer sourced from that garbled content — with nothing in the logs or the UI pointing back to "this was actually an OCR quality problem" (Part 11).
8. **A comparison question spanning two documents produces an answer that quietly mis-attributes a fact from Document B to Document A**, because the `compare` prompt never explicitly groups chunks by source and MMR/reranking can interleave them (D14) — subtle, hard to catch without the Part 20 evaluation framework in place.
9. **Hundreds of concurrent chat requests each spin up their own multi-thread pools for query-variant retrieval fan-out**, and the backend's OS thread count climbs into the thousands under a genuine peak-load event, degrading performance for everyone simultaneously rather than failing predictably for a few (D5).
10. **A student, deliberately or not, states something false in chat**, and the memory system stores it as a persistent "fact" that colors every subsequent answer in that same conversation, with nothing distinguishing "the student said this" from "a document confirmed this" (Part 17, memory poisoning).
