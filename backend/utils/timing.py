"""
timing.py

Per-request latency profiler for the /api/chat pipeline.

Usage:
    timing.start("chat:<conversation_id>")     # once, at the top of the request
    with timing.stage("embedding_generation"):
        ...
    log.info(timing.finish())                   # once, at the end of the request

Design notes:
- One `RequestTimer` per in-flight request, held in a `contextvars.ContextVar`
  so it is visible from every module the request touches (agent, rag_service,
  memory) without threading an explicit parameter through every function
  signature.
- `asyncio.to_thread` (used by routes/chat.py) copies the current context into
  its worker thread automatically, so the timer survives that hop. The
  `ThreadPoolExecutor`s used inside rag_service.py for concurrent Qdrant
  searches / LLM calls do NOT do this by default — `run_concurrent_ctx()`
  below is a drop-in replacement for a plain `pool.submit(fn)` that fixes
  that, so stage() calls made from those worker threads still land on the
  same request's timer instead of silently no-op'ing.
- `stage()` records a top-level, ordered entry (shown as its own row, summed
  if the same name is hit more than once in one request — e.g. one row per
  ReAct planning iteration). `substage()` records time into a named bucket
  WITHOUT adding an extra top-level row — used for costs that are already
  inside a wider stage's wall-clock window (e.g. cumulative embedding-model
  compute time inside the "retrieval" stage, where several variants are
  embedded concurrently so their individual durations cannot simply be
  summed into wall time).
- Overhead is a handful of `time.perf_counter()` calls per request — negligible.
"""

from __future__ import annotations

import contextvars
import logging
import time
from contextlib import contextmanager
from typing import Callable, List, Optional, TypeVar

log = logging.getLogger("profiling")

T = TypeVar("T")


class RequestTimer:
    def __init__(self, label: str = ""):
        self.label = label
        self._start = time.perf_counter()
        self._start_wall = time.time()
        self.stages: List[tuple] = []     # [(name, ms), ...] in call order (top-level only)
        self.notes: dict = {}             # name -> cumulative ms (top-level stages + substages)
        self.marks: dict = {}             # name -> elapsed ms since start, at a single instant

    def record_stage(self, name: str, ms: float) -> None:
        self.stages.append((name, ms))
        self.notes[name] = self.notes.get(name, 0.0) + ms

    def record_substage(self, name: str, ms: float) -> None:
        self.notes[name] = self.notes.get(name, 0.0) + ms

    def mark(self, name: str) -> float:
        """
        Record a point-in-time marker (elapsed ms since this timer's
        start) — for events that are a single INSTANT rather than a
        measured code block (e.g. "first streamed token was ready to
        send"), which stage()/substage() aren't a good fit for since they
        time a `with`-block's duration, not a point in time. Callers
        derive their own durations from pairs of marks (e.g.
        marks["stream_end"] - marks["first_token"]) — this method only
        ever does a plain dict assignment (safe for concurrent calls with
        DIFFERENT names from different threads, same reasoning as
        record_stage/record_substage; do not call it with the SAME name
        from two threads at once). Returns the recorded elapsed ms.
        """
        elapsed = self.total_ms()
        self.marks[name] = elapsed
        return elapsed

    def total_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000

    def report(self) -> str:
        total = self.total_ms()
        lines = [f"[profile] {self.label}  (started {time.strftime('%H:%M:%S', time.localtime(self._start_wall))})"]

        top_level_names = {n for n, _ in self.stages}
        for name, ms in self.stages:
            pct = (ms / total * 100) if total else 0.0
            lines.append(f"    {name:<38} {ms:9.1f} ms  ({pct:5.1f}%)")

        substage_only = {k: v for k, v in self.notes.items() if k not in top_level_names}
        if substage_only:
            lines.append("    -- sub-costs already inside a stage above --")
            for name, ms in sorted(substage_only.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {name:<38} {ms:9.1f} ms  (cumulative, may overlap concurrently)")

        accounted = sum(ms for _, ms in self.stages)
        unaccounted = total - accounted
        lines.append(f"    {'(unaccounted: planning/glue/serialization)':<38} {unaccounted:9.1f} ms")
        lines.append(f"    {'TOTAL':<38} {total:9.1f} ms")
        return "\n".join(lines)


_current: contextvars.ContextVar[Optional[RequestTimer]] = contextvars.ContextVar(
    "_current_request_timer", default=None
)


def start(label: str = "") -> RequestTimer:
    t = RequestTimer(label)
    _current.set(t)
    return t


def current() -> Optional[RequestTimer]:
    return _current.get()


@contextmanager
def stage(name: str):
    t = _current.get()
    if t is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        t.record_stage(name, (time.perf_counter() - t0) * 1000)


@contextmanager
def substage(name: str):
    t = _current.get()
    if t is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        t.record_substage(name, (time.perf_counter() - t0) * 1000)


def finish() -> Optional[str]:
    t = _current.get()
    if t is None:
        return None
    report = t.report()
    _current.set(None)
    return report


def run_concurrent_ctx(tasks: List[Callable[[], T]]) -> List[T]:
    """
    Same contract as rag_service._run_concurrent (run independent blocking
    callables on a small thread pool, return results in order), but submits
    each task wrapped in its OWN copy of the current context so
    stage()/substage() calls made inside those worker threads still land on
    the request's RequestTimer instead of silently becoming no-ops (plain
    ThreadPoolExecutor.submit does not propagate contextvars on its own).

    A fresh copy_context() per task is required, not one shared copy — a
    single Context object can only be entered (ctx.run(...)) by one thread
    at a time; sharing it across concurrently-submitted tasks raises
    "cannot enter context: ... already entered".
    """
    from concurrent.futures import ThreadPoolExecutor

    if not tasks:
        return []
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = [pool.submit(contextvars.copy_context().run, t) for t in tasks]
        return [f.result() for f in futures]
