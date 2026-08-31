"""
device.py

Shared torch device resolution for every local ML model in this codebase
(the sentence-transformers embedding model, the cross-encoder reranker).

One setting, EMBEDDING_DEVICE ("auto" / "cpu" / "cuda"), controls both —
they're both small local torch models loaded the same way, so a single
knob is simpler than two. "auto" (the default) uses CUDA when a GPU is
available and silently falls back to CPU otherwise, so this is always
safe to leave on default: it works unchanged on a machine with no GPU, or
with a CPU-only torch build (`torch.cuda.is_available()` just returns
False in that case — it never raises).

Measured impact (see PROFILING.md): on a CUDA-enabled torch build with an
RTX 3050, the multilingual-e5-large embedding batch used by MMR
diversification (~24 short passages) dropped from ~1.7-1.9s on CPU to
~0.17s on GPU (~10x); the cross-encoder reranker dropped from ~160ms to
~25-30ms warm (~6x). Embedding compute was the single largest contributor
to /api/chat latency (see profiling), so this directly targets the
measured bottleneck rather than a guess.
"""

from __future__ import annotations

import logging

from config import settings

log = logging.getLogger("device")

_resolved: str | None = None


def resolve_device() -> str:
    """
    Resolve settings.EMBEDDING_DEVICE to an actual torch device string
    ("cpu" or "cuda"), logging the decision once (cached after the first
    call — every local model in the process shares one decision).
    """
    global _resolved
    if _resolved is not None:
        return _resolved

    requested = (settings.EMBEDDING_DEVICE or "auto").strip().lower()

    import torch

    cuda_available = torch.cuda.is_available()

    if requested == "cpu":
        _resolved = "cpu"
    elif requested == "cuda":
        if cuda_available:
            _resolved = "cuda"
        else:
            log.warning(
                "EMBEDDING_DEVICE=cuda was set but no CUDA GPU is available "
                "(no GPU, missing driver, or a CPU-only torch build) — "
                "falling back to CPU."
            )
            _resolved = "cpu"
    else:
        if requested != "auto":
            log.warning(f"Unrecognized EMBEDDING_DEVICE={requested!r} — treating as 'auto'.")
        _resolved = "cuda" if cuda_available else "cpu"

    if _resolved == "cuda":
        log.info(f"Local ML models will run on CUDA ({torch.cuda.get_device_name(0)}).")
    else:
        log.info(
            "Local ML models will run on CPU "
            f"(EMBEDDING_DEVICE={requested!r}, cuda_available={cuda_available})."
        )

    return _resolved
