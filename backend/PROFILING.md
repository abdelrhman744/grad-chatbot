# Performance & Cross-Language Retrieval Investigation

This document records the investigation into two reported issues — slow
`/api/chat` responses, and failed Arabic→English (and untested
English→Arabic) cross-language retrieval — the evidence collected, the
root causes found, the fixes applied, and how to verify them.

All numbers below are real measurements taken by running the actual
backend against the real Qdrant collection and the real Groq API (not
synthetic/mocked), instrumented with the profiler added in this change
(`utils/timing.py`). Test machine: Windows 11 laptop, Intel CPU (14
logical threads), NVIDIA RTX 3050 Laptop GPU (4GB VRAM), running under
normal daily background load (browser, IDEs, Docker Desktop, etc. — see
"A note on measurement noise" below).

## How to reproduce this profiling yourself

Every `/api/chat` request now logs a per-stage timing breakdown
automatically (`LOG_REQUEST_PROFILE=true`, the default — see
`config.py`). To also get the verbose cross-language debug trail (query
variants, per-variant retrieved chunks, scores before/after rerank, final
LLM context), set `LOG_RETRIEVAL_DEBUG=true` in `backend/.env` — it's
`false` by default because it's verbose and meant for debugging, not
routine operation.

---

## Issue 1 — Why `/api/chat` is slow

### Instrumentation added

`utils/timing.py` adds a per-request `RequestTimer`, propagated via
`contextvars` (including through the `ThreadPoolExecutor`s used for
concurrent Qdrant searches / Groq calls — see `run_concurrent_ctx`, which
`rag_service._run_concurrent` now uses instead of a bare
`ThreadPoolExecutor`, since contextvars are not propagated to worker
threads by default and a naive shared-context fix crashes — see the
docstring). Every request logs a breakdown like this (real capture, see
below), stage timings and a `TOTAL`, to the `routes.chat` logger.

### Real measured breakdown (single English question, GPU enabled, "warm" server)

```
[profile] POST /api/chat conversation='verify_warm' query='What are the four main approaches to AI?'
    language_detection                           0.0 ms  (  0.0%)
    memory_loading                               0.0 ms  (  0.0%)
    agent_planning                             563.9 ms  (  2.4%)
    query_variant_generation_total            1180.4 ms  (  5.1%)
    embedding_and_qdrant_retrieval             954.2 ms  (  4.1%)
    reranking                                19223.6 ms  ( 83.4%)   <-- dominant cost
    memory_loading                               0.0 ms  (  0.0%)
    agent_planning                             692.4 ms  (  3.0%)
    llm_generation                             404.2 ms  (  1.8%)
    memory_persist                               0.0 ms  (  0.0%)
    -- sub-costs already inside "reranking" above --
    embedding_model_compute                  30063.5 ms  (cumulative across threads)
    mmr_diversification                      17191.7 ms  (cumulative)
    cross_encoder_inference                   1953.7 ms  (cumulative)
    TOTAL                                    23052.9 ms
```

This pattern (reranking = 78–91% of total, dominated by
`mmr_diversification`) was reproduced consistently across many separate
requests, on both CPU and GPU, and via a direct Python probe that calls
`rag_service._retrieve()` with no HTTP/ASGI layer involved at all — so
it's not an artifact of FastAPI/uvicorn.

### What each stage actually is (mapping the requested categories onto the real code)

The requested breakdown (language detection / query rewriting /
translation / query expansion / embedding / Qdrant retrieval / reranking
/ MMR / agent planning / memory / LLM generation) doesn't map 1:1 onto
separate serial steps in the current implementation — some of those are
deliberately fused or run concurrently:

- **`language_detection`** — regex-based (`detect_language`), effectively
  free (<1ms).
- **`agent_planning`** — the ReAct agent's tool-routing LLM call
  (`AGENT_MODEL=llama-3.1-8b-instant`), one call per iteration (typically
  2: retrieve, then generate). Cheap (~0.5-1s each) but adds up, and is a
  full Groq network round-trip each time.
- **`query_variant_generation_total`** — **query rewriting, query
  expansion, and translation are NOT three separate stages.**
  `_rewrite_query()` does spelling-correction AND synonym/concept
  expansion in **one** combined LLM call (by design, to cut Groq
  round-trips — see its docstring), and translation-to-English and
  translation-to-Arabic run **concurrently** with that same call in "wave
  1" (`_query_variants` in `rag_service.py`). A second wave then rewrites
  each translation. Total: 2 concurrent LLM waves, ~1-3s.
- **`embedding_and_qdrant_retrieval`** — embedding every query variant +
  searching Qdrant for each. **Originally** this fired one
  `embed_query()` call **per variant, concurrently, in its own thread**
  (up to ~9-15 variants). This turned out to be a real problem — see
  "A load-bearing side-finding" below — and was changed to one **batched**
  embedding call for all variants, then only the (cheap) Qdrant lookups
  are parallelized.
- **`reranking`** — cross-encoder scoring + lexical scoring + **MMR
  diversification** (all inside `_rerank()` / `_diversify()` in
  `rag_service.py`). This is the dominant cost — see below.
- **`memory_loading` / `memory_persist`** — in-RAM short-term memory +
  rendering the long-term fact store to text. Effectively free (0ms in
  every measurement) — long-term fact **extraction** (the only part of
  memory that calls an LLM) runs on a background thread and never blocks
  the response (see `memory_manager.py::_summarize_async`), so memory is
  **not** a bottleneck, confirmed, not assumed.
- **`llm_generation`** — the final answer-generation Groq call
  (`GROQ_MODEL`, the large 70B model). Consistently fast, 400-550ms —
  Groq's inference hardware is fast; this was never a concern.

### The bottleneck: MMR diversification's embedding call

`_rerank()` reranks the ~40-100 candidate chunks gathered across all
query variants, then (`RERANK_DIVERSIFY=true`, the default) reselects the
top N via a greedy MMR pass (`_diversify()`), which needs one **batched**
`embeddings.embed_documents()` call over the candidate shortlist (default
up to `max(RERANK_TOP_N*4, 12)` = 24 chunks) to compute chunk-to-chunk
similarity. This single call was consistently 60-85% of total request
time, every time, regardless of which of the two documents/languages was
queried.

**Why:** `intfloat/multilingual-e5-large` is a 560M-parameter transformer.
An isolated, clean benchmark of the exact same call (24 short passages,
same machine) measured:

| | CPU | GPU (RTX 3050) |
|---|---|---|
| embed_documents, batch of 24 | ~1.7–1.9s | ~0.17s (**~10x faster**) |
| cross-encoder, 24 pairs | ~160ms | ~25-30ms warm (**~6x faster**) |

This is a real, substantial, measured cost from a genuinely large model
doing genuinely heavy batched tensor computation — not a vague "CPU is
slow" hand-wave. It is the single largest, most consistent line item in
every real request measured.

### A load-bearing side-finding: concurrency hurts, doesn't help, this workload

The retrieval code fires one embedding call **per query variant,
concurrently** (`ThreadPoolExecutor`, up to ~9-15 threads). This pattern
is fine on CPU (independent cores), but was measured to make things
**worse**, not better, especially on GPU:

```
9 SEQUENTIAL single-item GPU embeds: 350.2 ms total
9 CONCURRENT single-item GPU embeds: wall=469.4ms, per-call 380-455ms each (!)
1 BATCHED call for the same 9 texts:  40.0 ms   <-- ~10x faster than concurrent
```

A single GPU serializes concurrently-submitted work rather than
parallelizing it the way independent CPU cores do, so N threads each
calling `.encode()` on one GPU is close to the worst possible pattern —
all of the per-call overhead, none of the batching benefit. **Fixed**:
`embeddings_provider.py` gained `embed_queries()` (a batched, multi-text
version of `embed_query()`), and `rag_service._retrieve()` now embeds all
query variants in **one** call up front, then only fans the (cheap,
I/O-bound) Qdrant lookups across threads — see `_search_by_vector()` in
`rag_service.py`. This is a straight improvement on both CPU and GPU, and
is what actually lets GPU acceleration pay off in this codebase's calling
pattern (see the GPU section below for why this matters more than it
sounds).

### A separate, real, environment-level cost: Groq API rate limiting

During testing, real `429 Too Many Requests` responses were hit from the
Groq API, with the SDK's own automatic retry-with-backoff adding **16-38
second delays** on top of an otherwise-normal request:

```
httpx | HTTP Request: POST .../chat/completions "HTTP/1.1 429 Too Many Requests"
groq._base_client | Retrying request to /openai/v1/chat/completions in 38.000000 seconds
```

This is outside the retrieval/reranking pipeline entirely — every LLM
call in a request (agent planning x2, query rewrite/translate x2 waves,
final generation) can independently hit this. Under sustained or bursty
traffic against a rate-limited Groq tier, this can dominate total latency
unpredictably and is worth addressing operationally (higher-tier Groq
rate limits, client-side request queuing/throttling, or caching
identical/near-identical planner calls) — it is not something the RAG
pipeline itself can fix.

### A note on measurement noise

This dev machine was running substantial background load throughout
testing (Adobe Creative Cloud, Discord, multiple VS Code windows, Docker
Desktop, a game launcher, several browser processes), confirmed via
`Get-Process`. This inflates and adds variance to every local-model
timing captured here, on both CPU and GPU — a clean/dedicated/production
machine should track closer to the isolated micro-benchmark numbers
above than the noisier "live pipeline" numbers. The GPU vs. CPU
*relative* findings (batching wins, MMR dominates, GPU power-state
matters — see below) are trustworthy; the absolute millisecond figures
from live end-to-end requests on this machine should be treated as
upper bounds, not a clean baseline.

---

## GPU acceleration

### Measure → identify → estimate → implement, in that order

1. **Measured current CPU performance**: see the table above — ~1.7-1.9s
   per MMR embedding batch, ~160ms cross-encoder, both on CPU, in a clean
   isolated benchmark (`torch.get_num_threads()` = 14 on this machine).
2. **Identified the actual bottleneck**: MMR diversification's batched
   embedding call, 60-85% of total request time, confirmed via the
   profiler across many real requests plus a no-HTTP direct probe.
3. **Confirmed a GPU was actually available** before assuming anything:
   `nvidia-smi` showed an RTX 3050 Laptop GPU (4GB VRAM) on this machine.
   The installed `torch` was a CPU-only build (`2.13.0+cpu`) — GPU was
   not being used at all prior to this change.
4. **Measured (not estimated) the expected speedup** by installing a
   CUDA-enabled torch build (`torch==2.13.0+cu126`,
   `pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.13.0`)
   and re-running the exact same benchmark: **~10x** for the embedding
   model, **~6x** for the cross-encoder (table above). This justified
   implementing GPU support.

### Implementation

- `utils/device.py`: `resolve_device()` — one shared device decision
  (`settings.EMBEDDING_DEVICE`, default `"auto"`) for both local torch
  models. `"auto"` uses CUDA when `torch.cuda.is_available()` is true,
  else CPU. Never raises: a CPU-only torch build, or a machine with no
  GPU, works exactly as before (`torch.cuda.is_available()` just returns
  `False`). Force with `EMBEDDING_DEVICE=cpu` or `EMBEDDING_DEVICE=cuda`
  (the latter logs a warning and falls back to CPU if no GPU is actually
  available, rather than crashing).
- `services/embeddings_provider.py`: `LocalEmbeddings` now loads its
  `SentenceTransformer` with `device=resolve_device()`.
- `services/rag_service.py`: `_get_cross_encoder()` now loads the
  `CrossEncoder` with the same `device=resolve_device()`.
- No other code changes needed for GPU support itself — both wrap the
  same `sentence-transformers` classes either device already supported.

### Configuration / new dependencies

See `backend/.env.example` (`EMBEDDING_DEVICE`) and
`backend/requirements.txt` (comment above the `torch` line). Summary:

- Default `requirements.txt` install (`pip install -r requirements.txt`)
  installs the CPU-only `torch` build — works everywhere, no change in
  behavior for anyone who doesn't opt in.
- To actually use a GPU, install a CUDA build of `torch` **instead**,
  matching your NVIDIA driver, e.g.:
  ```
  pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.13.0
  ```
  (pick the `cuXXX` tag for your driver — see
  https://pytorch.org/get-started/locally/). `EMBEDDING_DEVICE=auto` (the
  default) picks this up automatically; no other config change needed.

### An important, honest caveat found while verifying the fix

The clean isolated benchmark (~10x/~6x speedup) did **not** fully
materialize in live, single-request testing on this machine. Investigated
with real telemetry rather than guessing:

```
nvidia-smi --query-gpu=clocks.current.sm,pstate,clocks_event_reasons.active
210 MHz, P8, 0x...01 (idle)   <- most of the request: GPU asleep, waiting on Groq/CPU work
...
1957 MHz, P0, 0x...00 (busy)  <- brief burst: GPU actually computing
```

The RTX 3050 **Laptop** GPU, under Windows' default power management,
drops to its lowest power state (P8, ~210MHz) whenever idle, and this
app's workload is bursty — short GPU calls interleaved with
network-bound Groq calls and Qdrant lookups — so the GPU frequently has
time to fall back asleep between bursts. Each new burst pays a real
wake-up/ramp-to-boost-clock cost (210MHz → 1957MHz is roughly the ~10x
gap between the clean sustained benchmark and the live bursty numbers).
This is a genuine property of this deployment environment (laptop GPU +
Windows power management + low-frequency bursty calls), **not a defect
in the implementation** — the device selection and computation are
correct and verified (GPU utilization and `nvidia-smi`'s process list
both confirm the GPU is genuinely being used).

**Practical implication**: on a server-class GPU, a machine with the
NVIDIA power plan set to "Prefer Maximum Performance" instead of the
default adaptive/optimal-power mode, or under sustained/higher-throughput
traffic (which keeps the GPU warm across requests instead of letting it
idle between them), the full ~10x/~6x speedup should be realized
consistently, matching the clean benchmark. On this specific
low-traffic, power-managed dev laptop, GPU and CPU were measured to
perform similarly for a single isolated request — **the batching fix
above (one call instead of N concurrent calls) is the change that helps
unconditionally, on both CPU and GPU, regardless of this power-state
effect**, and is the one to trust as an unambiguous win. Verify on your
actual target hardware before concluding GPU is or isn't worth enabling
for your deployment.

---

## Issue 2 — Cross-language retrieval

### Method: instrument first, then test each direction directly

`settings.LOG_RETRIEVAL_DEBUG` (off by default) logs, for every retrieval:
original query, detected language, every generated query variant, every
variant's retrieved chunks + raw vector-similarity score ("before
rerank"), the blended score after rerank, and (in `generate_answer`) the
final context text actually sent to the LLM. This was used to answer
every sub-question the investigation asked for.

To isolate the retrieval pipeline from the agent's routing layer (see
below — they turned out to be two different things), `rag_service._retrieve()`
was also called **directly**, bypassing the HTTP/agent layer entirely, via
a standalone script.

### Finding #1: the deterministic retrieval pipeline is correct

Direct probe, Arabic question about a chunk that only exists in the
English-language PDF:

```
QUERY (ar): ما هي عناصر إطار PEAS؟   ("What are the elements of the PEAS framework?")
-> 4 docs returned, all from the English PDF
Rank 1: ...report.pdf p8 score=0.8796 | 26 (pp. 26-32) 27 (pp. 26-32) ...
Rank 2: ...report.pdf p3 score=0.7796 | Four main approaches to AI ...
```

Walking through every sub-question asked:

- **Is the Arabic query translated to English?** Yes —
  `_query_variants()` runs `_translate(q, "en")` and `_translate(q, "ar")`
  concurrently with query rewriting ("wave 1"), confirmed via debug logs
  showing the English-translated variant among the generated variants.
- **Is expansion before or after translation?** Neither strictly — they
  run **concurrently** in wave 1 (rewriting+expansion of the *original*
  query happens alongside, not before/after, both translation
  directions); a **second** wave then rewrites/expands each translation
  too (`wave2` in `_query_variants`). So expansion happens on both the
  original-language query and on each translated variant.
- **Are both Arabic and English variants actually searched?** Yes —
  confirmed via debug logs showing per-variant Qdrant hit counts for
  every one of ~10-13 variants (original, normalized, translated,
  rewritten, expanded — up to 22, deduplicated).
- **Are all variants sent to Qdrant?** Yes, all deduplicated variants.
- **Are retrieved chunks merged correctly?** Yes —
  `_deduplicate_retrieved()` merges by (content-prefix, source, page)
  across all variants' results before reranking.
- **Is reranking discarding the English chunks?** No — in every direct
  test, the English PDF's chunks ranked at the top after reranking
  (scores 0.65-0.88), not discarded.
- **Is the confidence threshold filtering correctly?** Yes —
  `CONFIDENCE_THRESHOLD=0.05` default is far below the observed top
  scores (0.65+); it did not reject valid cross-language results in any
  test.
- **Does the final prompt receive the English context?** Yes — logged
  and confirmed the exact context chunks (English PDF text) passed to
  `build_prompt_with_memory()`.

**Conclusion: the retrieval pipeline itself was not the bug.** This was
verified, not assumed — via direct code-level testing bypassing the HTTP
layer, with full before/after-rerank score logging as evidence.

### Finding #2: the actual failure is in the agent's tool-routing step

Testing the **same exact Arabic question through the real `/api/chat`
HTTP endpoint** (full agent loop) reproduced a real failure:

```
POST /api/chat {"query": "اشرح لي إطار PEAS", ...}
-> "This system only answers questions about the uploaded documents.
    We haven't discussed any documents yet..."
```

— returned in ~1 second, meaning **retrieve was never called at all.**
The `ExecutionContext` had zero documents and zero retrieval attempts;
the agent's small, low-latency routing model
(`AGENT_MODEL=llama-3.1-8b-instant`) chose the `respond` action directly.

Isolating the routing decision itself (calling `AgentLLM.invoke()`
directly, same exact prompt, `temperature=0.0`) showed this is
**intermittent, not deterministic**: 3 repeated calls with the identical
input all correctly chose `retrieve`, while the earlier live HTTP request
with the exact same text chose `respond`. Hosted LLM inference (Groq, and
every other major provider) is not perfectly reproducible run-to-run even
at temperature 0 — floating-point non-associativity across different
GPU/inference batches means the same prompt can occasionally cross a
decision boundary differently. The system prompt (`agent/prompt.py`)
already states the correct rule explicitly ("HARD RULE: you may NEVER
choose respond ... before at least one retrieve call has been made this
turn, UNLESS pure greeting/small talk") — but it was **only** enforced by
instruction, never by code, so any small-model slip became a direct
user-facing failure. This reproduced more easily for short, imperative
Arabic phrasing ("اشرح لي X" — "explain X to me", no question mark) than
for clearly-interrogative phrasing ("ما هي X؟" — "what is X?"), though the
sample size here is small (a handful of trials) — this is a plausible
contributing factor, not a certainty, and is the kind of thing worth
watching for in production logs (the new `agent_planning` profiling stage
and the `WARNING` log added below both surface it going forward).

### Root cause

**Not** a retrieval-pipeline defect. It's LLM sampling variance in the
agent's routing step occasionally violating a rule the system prompt
already states but never enforced deterministically.

### Fix

`agent/agent.py`: added `Agent._correct_premature_terminal()`, called
right after every planning decision in both `run()` and `run_stream()`.
If the planner chooses `respond` or `generate` **and nothing has been
retrieved yet this turn** (`context.retrieved_questions` and
`context.documents` both empty), it does **not** immediately accept that
action — it re-asks the planner **once**, appending a corrective message
that restates the prompt's own rule and explicitly calls out short
imperative phrasings as information requests. Whatever the second answer
is gets accepted (including `respond` again, if it's genuinely a
greeting) — this never *forces* retrieval, it only asks the model to
double-check itself against a rule it already knows, which is enough to
correct the observed failure mode without breaking legitimate small talk.
Logs a `WARNING` each time it fires, so this can be monitored in
production (a high fire rate would indicate the underlying model/prompt
needs more direct attention, not just this safety net).

### Verification: before vs. after

**Before the fix**, live HTTP request:
```
"اشرح لي إطار PEAS" -> "This system only answers questions about the
uploaded documents..." (retrieve never called)
```

**After the fix**, same exact request, same conversation state, log shows
the backstop catching the exact same planner mistake and correcting it:
```
WARNING | agent | Agent chose 'respond' on iteration 1 with nothing
retrieved yet for question='اشرح لي إطار PEAS' — re-asking once per the
prompt's own HARD RULE.
-> retrieved English PDF chunks, answered correctly:
"PEAS إطار يتكون من: Performance measure, Environment, Actuators, و Sensors."
sources: Lecture 1&2 (...AI...)_report.pdf (p. 3, 4, 6)
```
Reproduced across 2 separate trials — the backstop fired both times (the
underlying flakiness is real and recurring) and corrected it both times.

**Arabic question → English-only document** ✅ (verified above, both via
direct pipeline probe and full HTTP agent flow).

**English question → Arabic-only document** ✅ — ingested a small
Arabic-only test document (`arabic_leave_policy_TEST.txt`, a fabricated
company leave policy stating "30 يومًا إجازة سنوية مدفوعة الأجر" / "30 days
paid annual leave", not present in any other document), then asked in
English:
```
POST /api/chat {"query": "How many paid annual leave days does an
employee get?"}
-> {"answer": "30",
    "sources": "... | arabic_leave_policy_TEST.txt (p. 1)"}
```
Correct answer, correct source, cross-language in the other direction.

---

## Summary of code changes

| File | Change |
|---|---|
| `utils/timing.py` (new) | Per-request stage profiler, contextvar-propagated across thread pools |
| `utils/device.py` (new) | Shared CPU/CUDA device resolution (`auto`/`cpu`/`cuda`) |
| `routes/chat.py` | Starts/logs the profiler per request |
| `agent/agent.py` | Stage timers; **`_correct_premature_terminal()`** — the Issue 2 fix |
| `services/rag_service.py` | Stage timers; retrieval debug logging; **batched query-variant embedding** (was N concurrent single calls); cross-encoder loads on `resolve_device()` |
| `services/embeddings_provider.py` | Loads on `resolve_device()`; added `embed_queries()` (batched) |
| `config.py` / `.env.example` | `LOG_REQUEST_PROFILE`, `LOG_RETRIEVAL_DEBUG`, `EMBEDDING_DEVICE` |
| `requirements.txt` | Documented the optional CUDA-torch install (no default behavior change) |

## What's next (not done here — out of scope for "measure and fix", flagged for follow-up)

- The Groq 429 rate-limiting is a real, separate latency source under
  load; worth addressing operationally (see Issue 1).
- `agent_planning`'s 2+ sequential Groq round-trips per request are a
  smaller but real cost; a cheaper/deterministic first-pass router (or
  fewer max ReAct iterations) could reduce this further.
- MMR diversification's embedding cost, while now batched (helping both
  CPU and GPU), is still the largest remaining line item; consider
  caching per-chunk embeddings (they don't change between requests) so
  diversification only needs to embed once per document at ingestion
  time, not per query.
