"""
evaluate_agent_token_and_quality.py

Live token-consumption + quality audit for the agent/LLM pipeline (Task:
"Audit token consumption and switch the default model to
qwen/qwen3.8-27b"). Not a mock/synthetic benchmark — it runs the REAL
`Agent.run()` path (agent/session.py -> agent/agent.py) against the real
Groq API and the real local Qdrant/embeddings stack, exactly like a live
`/api/chat` request would.

What it measures per scenario:
- Every real Groq API call made (model, prompt_tokens, completion_tokens,
  total_tokens, from the API's own `usage` field — not estimated), via a
  transparent monkeypatch of `groq`'s low-level `Completions.create`
  (services/llm_provider.py is the only thing that calls it, so this
  covers every call site: agent planning, query rewrite/translate, answer
  generation, summarize/compare, memory-only respond).
- Wall-clock latency for the whole turn.
- A cheap heuristic quality check appropriate to the scenario type
  (grounded questions should NOT contain the "not available" refusal;
  ungrounded questions SHOULD; small talk should not trigger retrieval).

Scenario matrix: {en, ar} x {content, small_talk} x {turn1, turn2}, plus
one ungrounded content question per language — see SCENARIOS below.

Usage:
    cd backend
    python scripts/evaluate_agent_token_and_quality.py --label before
    # edit .env / restart nothing needed other than re-running this
    # script AFTER changing backend/.env, since model selection is
    # captured into module-level singletons at import time (same as the
    # real app) -- a fresh process is required to pick up a new .env,
    # exactly like restarting the backend would be.
    python scripts/evaluate_agent_token_and_quality.py --label after

Each run resets its own dedicated test conversation_ids first (fresh
Qdrant points + fresh memory_storage file) so `--label before` and
`--label after` runs never leak state into each other, then re-ingests a
small fixed EN/AR test corpus. Results are written as JSON to
scripts/_token_audit_results/<label>.json for later comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Written outside the repo by default (same convention as
# scripts/evaluate_ocr_followup.py's OCR_EVAL_SCRATCH) so ad-hoc audit runs
# don't pollute `git status` — override with TOKEN_AUDIT_SCRATCH if needed.
RESULTS_DIR = Path(os.environ.get(
    "TOKEN_AUDIT_SCRATCH",
    str(Path.home() / "AppData/Local/Temp/claude/c--Graduation-Agentic-AI-grad-chatbot"
        "/d6ce649c-e829-4a89-8143-55d2f1380fa8/scratchpad/token_audit"),
))

# ── Transparent Groq call interceptor (must be installed BEFORE any app
#    module builds its shared GroqLLM/client singletons) ───────────────────
import groq.resources.chat.completions as _groq_completions  # noqa: E402

_CALL_LOG: list[dict] = []
_orig_create = _groq_completions.Completions.create


def _patched_create(self, *args, **kwargs):
    t0 = time.time()
    resp = _orig_create(self, *args, **kwargs)
    elapsed_ms = (time.time() - t0) * 1000
    usage = getattr(resp, "usage", None)
    _CALL_LOG.append({
        "model": kwargs.get("model"),
        "json_mode": bool(kwargs.get("response_format")),
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        "latency_ms": round(elapsed_ms, 1),
    })
    return resp


_groq_completions.Completions.create = _patched_create

# ── Now safe to import the app ──────────────────────────────────────────────
from config import settings  # noqa: E402
from services import rag_service  # noqa: E402
from agent.session import get_agent, _agents, _lock  # noqa: E402
from utils import timing  # noqa: E402


def _percentile(values: "list[float]", pct: float) -> float:
    """Nearest-rank percentile over a small sample — no numpy/scipy
    dependency needed for ~10-40 data points. `pct` in [0, 100]."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(pct / 100 * (len(s) - 1))))
    return s[k]


EN_DOC = """Acme Cloud Storage — Data Retention Policy (Internal Reference)

Acme Cloud Storage retains deleted customer files for exactly 45 days in a
recovery bin before permanent erasure. Files larger than 2 GB are excluded
from the recovery bin and are erased immediately upon deletion. Enterprise
tier customers may request an extended recovery window of up to 90 days by
contacting account support. Backup snapshots of the recovery bin are taken
every 6 hours. The policy was last revised on March 3, 2025, by the Data
Governance team.
"""

AR_DOC = """شركة أكمي للتخزين السحابي — سياسة الاحتفاظ بالبيانات (مرجع داخلي)

تحتفظ شركة أكمي للتخزين السحابي بالملفات المحذوفة للعملاء لمدة 45 يومًا
بالضبط في سلة الاسترجاع قبل حذفها نهائيًا. الملفات التي يزيد حجمها عن 2
جيجابايت مستثناة من سلة الاسترجاع ويتم حذفها فورًا عند حذفها من قبل
المستخدم. يمكن لعملاء الفئة المؤسسية طلب فترة استرجاع ممتدة تصل إلى 90
يومًا عن طريق التواصل مع دعم الحسابات. يتم أخذ نسخ احتياطية من سلة
الاسترجاع كل 6 ساعات. تمت مراجعة هذه السياسة آخر مرة في 3 مارس 2025 من
قبل فريق حوكمة البيانات.
"""

TEST_CONVOS = {
    "en_content": "token_audit_en_content",
    "ar_content": "token_audit_ar_content",
    "en_smalltalk": "token_audit_en_smalltalk",
    "ar_smalltalk": "token_audit_ar_smalltalk",
    "en_ungrounded": "token_audit_en_ungrounded",
    "ar_ungrounded": "token_audit_ar_ungrounded",
}

NOT_AVAILABLE_MARKERS = [
    "not available in the uploaded files",
    "غير موجودة في الملفات المرفوعة",
    "only answers questions about the uploaded documents",
    "لا يجيب إلا عن أسئلة",
    # rag_service.generate_answer()'s early-return phrasing when retrieval
    # found genuinely zero documents (below CONFIDENCE_THRESHOLD) — a
    # DIFFERENT, equally-correct refusal path from build_prompt's
    # LLM-generated "not available" phrasing above (no LLM call at all in
    # this path — see generate_answer's `if not documents:` branch).
    "no relevant documents were found",
    "لا توجد مستندات كافية للإجابة",
]


def _reset_conversation(conversation_id: str) -> None:
    with _lock:
        _agents.pop(conversation_id, None)
    try:
        rag_service.delete_conversation_documents(conversation_id)
    except Exception as e:
        print(f"  (reset) delete_conversation_documents warning for {conversation_id}: {e}")
    from memory.summary_memory import SummaryMemory
    SummaryMemory().delete_facts(conversation_id)
    from memory.short_memory import ShortMemory  # noqa: F401  (just confirming import path)


def _ingest(conversation_id: str, filename: str, text: str) -> None:
    n = rag_service.update_db_files(
        [{"filename": filename, "data": text.encode("utf-8")}],
        conversation_id=conversation_id,
    )
    print(f"  ingested {filename} -> {n} chunk(s) into conversation_id={conversation_id!r}")


def _run_turn(conversation_id: str, question: str, language: str = "auto") -> dict:
    agent = get_agent(conversation_id)
    start_idx = len(_CALL_LOG)
    req_timer = timing.start(label=f"bench:{conversation_id}")
    t0 = time.time()
    context = agent.run(question, language=language)
    elapsed_ms = (time.time() - t0) * 1000
    timing.finish()
    calls = _CALL_LOG[start_idx:]
    answer = context.final_answer() or ""
    return {
        "question": question,
        "answer": answer,
        "num_documents_retrieved": len(context.documents),
        "elapsed_ms": round(elapsed_ms, 1),
        # Per-stage wall-clock breakdown (planner/retrieval/generation/...)
        # from the app's own profiler (utils/timing.py) — same stage names
        # PROFILING.md documents. Substage entries (e.g.
        # "query_rewrite_and_translate") overlap with their parent stage's
        # wall time, so don't sum stage_breakdown values expecting them to
        # equal elapsed_ms.
        "stage_breakdown_ms": {k: round(v, 1) for k, v in req_timer.notes.items()},
        "groq_calls": calls,
        "num_groq_calls": len(calls),
        "prompt_tokens_total": sum(c["prompt_tokens"] or 0 for c in calls),
        "completion_tokens_total": sum(c["completion_tokens"] or 0 for c in calls),
        "total_tokens_total": sum(c["total_tokens"] or 0 for c in calls),
    }


def _measure_ttft(conversation_id: str, question: str) -> dict | None:
    """
    Time-to-first-token for the terminal generation call, via the real
    streaming path (agent.run_stream) — the non-streaming benchmark above
    can't observe this (it only sees the finished string). Run as a
    one-off EXTRA turn (not part of the main scenario matrix, so it
    doesn't affect the turn1/turn2 token counts) against a
    conversation_id that already has documents ingested.
    """
    agent = get_agent(conversation_id)
    start_idx = len(_CALL_LOG)
    t0 = time.time()
    first_token_ms = None
    full_text = ""
    try:
        for event in agent.run_stream(question):
            if event.get("type") == "token" and first_token_ms is None:
                first_token_ms = (time.time() - t0) * 1000
            if event.get("type") == "token":
                full_text += event.get("text", "")
    except Exception as e:
        print(f"  !! TTFT probe failed: {e}")
        return None
    total_ms = (time.time() - t0) * 1000
    calls = _CALL_LOG[start_idx:]
    return {
        "question": question,
        "ttft_ms": round(first_token_ms, 1) if first_token_ms is not None else None,
        "total_stream_ms": round(total_ms, 1),
        "answer_preview": full_text[:160],
        "num_groq_calls": len(calls),
    }


def _is_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(m.lower() in low for m in NOT_AVAILABLE_MARKERS)


def build_scenarios() -> "list[dict]":
    scenarios = []

    # ── EN content (grounded), turn 1 + turn 2 ──────────────────────────
    cid = TEST_CONVOS["en_content"]
    scenarios.append({
        "id": "en_content_turn1_grounded",
        "conversation_id": cid,
        "question": "How many days does Acme keep deleted files in the recovery bin before permanent erasure?",
        "check": lambda r: (not _is_refusal(r["answer"])) and ("45" in r["answer"]),
    })
    scenarios.append({
        "id": "en_content_turn2_grounded",
        "conversation_id": cid,
        "question": "And what about the extended window for enterprise customers?",
        "check": lambda r: (not _is_refusal(r["answer"])) and ("90" in r["answer"]),
    })

    # ── AR content (grounded), turn 1 + turn 2 ──────────────────────────
    cid = TEST_CONVOS["ar_content"]
    scenarios.append({
        "id": "ar_content_turn1_grounded",
        "conversation_id": cid,
        "question": "كام يوم بتحتفظ أكمي بالملفات المحذوفة في سلة الاسترجاع قبل ما تتمسح نهائي؟",
        "check": lambda r: (not _is_refusal(r["answer"])) and ("45" in r["answer"]),
    })
    scenarios.append({
        "id": "ar_content_turn2_grounded",
        "conversation_id": cid,
        "question": "طيب وفترة الاسترجاع الممتدة لعملاء المؤسسات كام؟",
        "check": lambda r: (not _is_refusal(r["answer"])) and ("90" in r["answer"]),
    })

    # ── EN small talk, turn 1 + turn 2 (no retrieval expected) ──────────
    cid = TEST_CONVOS["en_smalltalk"]
    scenarios.append({
        "id": "en_smalltalk_turn1",
        "conversation_id": cid,
        "question": "Hi there!",
        "check": lambda r: r["num_documents_retrieved"] == 0,
    })
    scenarios.append({
        "id": "en_smalltalk_turn2",
        "conversation_id": cid,
        "question": "Thanks!",
        "check": lambda r: r["num_documents_retrieved"] == 0,
    })

    # ── AR small talk, turn 1 + turn 2 ───────────────────────────────────
    cid = TEST_CONVOS["ar_smalltalk"]
    scenarios.append({
        "id": "ar_smalltalk_turn1",
        "conversation_id": cid,
        "question": "مرحبا",
        "check": lambda r: r["num_documents_retrieved"] == 0,
    })
    scenarios.append({
        "id": "ar_smalltalk_turn2",
        "conversation_id": cid,
        "question": "شكرا",
        "check": lambda r: r["num_documents_retrieved"] == 0,
    })

    # ── Ungrounded content questions (doc exists, but doesn't cover this) ─
    cid = TEST_CONVOS["en_ungrounded"]
    scenarios.append({
        "id": "en_content_ungrounded",
        "conversation_id": cid,
        "question": "What is the CEO's name and what year was Acme founded?",
        "check": lambda r: _is_refusal(r["answer"]),
    })

    cid = TEST_CONVOS["ar_ungrounded"]
    scenarios.append({
        "id": "ar_content_ungrounded",
        "conversation_id": cid,
        "question": "مين الرئيس التنفيذي لشركة أكمي وفي أنهي سنة اتأسست؟",
        "check": lambda r: _is_refusal(r["answer"]),
    })

    return scenarios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="e.g. 'before' or 'after' — used as the output filename")
    parser.add_argument("--sleep-between", type=float, default=2.0, help="seconds to sleep between scenarios (rate-limit courtesy)")
    args = parser.parse_args()

    print(f"=== Token/quality audit run: label={args.label!r} ===")
    print(f"GROQ_MODEL={settings.GROQ_MODEL!r}  AGENT_MODEL={settings.AGENT_MODEL!r}  AGENT_FALLBACK_MODEL={settings.AGENT_FALLBACK_MODEL!r}")

    print("\n--- Resetting test conversations ---")
    for cid in TEST_CONVOS.values():
        _reset_conversation(cid)

    print("\n--- Ingesting test corpus ---")
    _ingest(TEST_CONVOS["en_content"], "acme_retention_policy_en.txt", EN_DOC)
    _ingest(TEST_CONVOS["ar_content"], "acme_retention_policy_ar.txt", AR_DOC)
    _ingest(TEST_CONVOS["en_ungrounded"], "acme_retention_policy_en.txt", EN_DOC)
    _ingest(TEST_CONVOS["ar_ungrounded"], "acme_retention_policy_ar.txt", AR_DOC)
    # smalltalk conversations deliberately get NO documents.

    print("\n--- Running scenarios ---")
    scenarios = build_scenarios()
    results = []
    for i, sc in enumerate(scenarios):
        print(f"[{i+1}/{len(scenarios)}] {sc['id']} ...")
        try:
            r = _run_turn(sc["conversation_id"], sc["question"])
        except Exception as e:
            print(f"  !! FAILED: {e}")
            results.append({"id": sc["id"], "error": str(e)})
            time.sleep(args.sleep_between)
            continue
        r["id"] = sc["id"]
        try:
            r["check_passed"] = bool(sc["check"](r))
        except Exception as e:
            r["check_passed"] = None
            r["check_error"] = str(e)
        results.append(r)
        print(
            f"  -> {r['num_groq_calls']} Groq call(s), "
            f"{r['prompt_tokens_total']} prompt / {r['completion_tokens_total']} completion tokens, "
            f"{r['elapsed_ms']:.0f}ms, check_passed={r.get('check_passed')}"
        )
        print(f"  answer preview: {r['answer'][:160]!r}")
        time.sleep(args.sleep_between)

    print("\n--- TTFT probe (streaming path, extra turn, not part of the matrix above) ---")
    ttft = _measure_ttft(
        TEST_CONVOS["en_content"],
        "What is the backup snapshot interval for the recovery bin?",
    )
    if ttft:
        print(f"  TTFT: {ttft['ttft_ms']}ms  |  full stream: {ttft['total_stream_ms']}ms  |  {ttft['num_groq_calls']} Groq call(s)")
        print(f"  answer preview: {ttft['answer_preview']!r}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.label}.json"

    turn_latencies = [r["elapsed_ms"] for r in results if "elapsed_ms" in r]
    call_latencies = [c["latency_ms"] for r in results for c in r.get("groq_calls", []) if c.get("latency_ms") is not None]

    # Aggregate stage-breakdown across all turns (sum per stage name) —
    # gives an at-a-glance planner-vs-retrieval-vs-generation split across
    # the whole run, same stage names PROFILING.md uses.
    stage_totals: dict[str, float] = {}
    for r in results:
        for k, v in r.get("stage_breakdown_ms", {}).items():
            stage_totals[k] = stage_totals.get(k, 0.0) + v

    summary = {
        "total_groq_calls": sum(r.get("num_groq_calls", 0) or 0 for r in results),
        "total_prompt_tokens": sum(r.get("prompt_tokens_total", 0) or 0 for r in results),
        "total_completion_tokens": sum(r.get("completion_tokens_total", 0) or 0 for r in results),
        "avg_turn_latency_ms": round(sum(turn_latencies) / len(turn_latencies), 1) if turn_latencies else 0.0,
        "p50_turn_latency_ms": round(_percentile(turn_latencies, 50), 1),
        "p95_turn_latency_ms": round(_percentile(turn_latencies, 95), 1),
        "avg_call_latency_ms": round(sum(call_latencies) / len(call_latencies), 1) if call_latencies else 0.0,
        "p50_call_latency_ms": round(_percentile(call_latencies, 50), 1),
        "p95_call_latency_ms": round(_percentile(call_latencies, 95), 1),
        "stage_totals_ms": {k: round(v, 1) for k, v in stage_totals.items()},
        "ttft_ms": ttft["ttft_ms"] if ttft else None,
        "checks_passed": sum(1 for r in results if r.get("check_passed") is True),
        "checks_failed": sum(1 for r in results if r.get("check_passed") is False),
        "checks_errored": sum(1 for r in results if "error" in r),
        "checks_total": len(results),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "label": args.label,
            "groq_model": settings.GROQ_MODEL,
            "agent_model": settings.AGENT_MODEL,
            "agent_fallback_model": settings.AGENT_FALLBACK_MODEL,
            "summary": summary,
            "ttft_probe": ttft,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n=== Summary (label={args.label!r}) ===")
    print(f"Total Groq calls: {summary['total_groq_calls']}")
    print(f"Total prompt tokens: {summary['total_prompt_tokens']}  |  Total completion tokens: {summary['total_completion_tokens']}")
    print(f"Turn latency  avg={summary['avg_turn_latency_ms']}ms  p50={summary['p50_turn_latency_ms']}ms  p95={summary['p95_turn_latency_ms']}ms  (n={len(turn_latencies)})")
    print(f"Call latency  avg={summary['avg_call_latency_ms']}ms  p50={summary['p50_call_latency_ms']}ms  p95={summary['p95_call_latency_ms']}ms  (n={len(call_latencies)})")
    print(f"TTFT: {summary['ttft_ms']}ms")
    print("Stage totals (ms, summed across all turns):")
    for k, v in sorted(stage_totals.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<32} {v:9.1f}")
    print(f"Checks passed: {summary['checks_passed']}/{summary['checks_total']}  failed: {summary['checks_failed']}  errored: {summary['checks_errored']}")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
