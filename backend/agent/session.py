"""
session.py

Keeps one Agent instance per conversation_id alive across requests, so
<<<<<<< HEAD
short-term memory (and the loaded long-term summary) persists between
turns without re-reading disk every time. This is an in-process registry;
it resets when the backend restarts (the persisted long-term summary on
disk survives restarts regardless).
=======
short-term memory persists between turns without being rebuilt from
scratch on every request. This is an in-process registry; it resets when
the backend restarts — conversational memory is short-term/in-RAM only
(see memory/memory_manager.py), so a restart clears it the same way idle
eviction below does.
>>>>>>> aac0288 (Initial commit - LASTVERSION)

Idle-Agent cleanup
------------------
Without cleanup, this registry grows by one entry per distinct
conversation_id FOREVER — every browser tab that has ever connected stays
in RAM even after the user closes it, which is an unbounded memory leak in
a long-running deployment (see the Agent-lifecycle investigation).

A daemon background thread (`_cleanup_loop`) periodically scans the
registry and evicts entries that are both idle (no activity for
`AGENT_IDLE_TIMEOUT_SECONDS`) AND not currently handling a request
(`Agent.is_idle()` — see agent/agent.py, which tracks its own in-flight
count so a slow request can never be evicted out from under itself). This
<<<<<<< HEAD
only frees RAM: it does not touch the persisted
`memory_storage/<conversation_id>.json` fact store, and a later request
for the same conversation_id transparently creates a fresh Agent (empty
short-term memory, but its long-term facts are reloaded from disk exactly
as they would be on a cold start today — see MemoryManager.__init__).
=======
only frees RAM. Conversational memory is short-term only (see
memory/memory_manager.py) — there is no persisted long-term store, so a
later request for the same conversation_id transparently creates a fresh
Agent with empty memory, exactly as it would on a cold start.
>>>>>>> aac0288 (Initial commit - LASTVERSION)
"""

from __future__ import annotations

import logging
import threading
import time

from config import settings

from .agent import Agent

log = logging.getLogger("agent.session")

_agents: dict[str, Agent] = {}
_lock = threading.Lock()


def get_agent(conversation_id: str = settings.DEFAULT_CONVERSATION_ID) -> Agent:
    with _lock:
        agent = _agents.get(conversation_id)
        if agent is None:
            agent = Agent(conversation_id=conversation_id)
            _agents[conversation_id] = agent
            log.info(
                f"Created Agent for conversation_id={conversation_id!r} "
                f"(active conversations in RAM: {len(_agents)})"
            )
        agent.touch()
        return agent


def reset_agent(conversation_id: str = settings.DEFAULT_CONVERSATION_ID) -> None:
    with _lock:
        agent = _agents.get(conversation_id)
        if agent:
            agent.reset_memory()
            agent.touch()


# ── Idle-Agent cleanup ───────────────────────────────────────────────────

def _cleanup_once() -> int:
    """
    One cleanup pass: evict every registry entry whose Agent reports
    `is_idle(...)`. Runs entirely under `_lock`, same as every other
    registry mutation (`get_agent`/`reset_agent`) — a request that's
    already holding a reference to its Agent (returned by a prior
    `get_agent()` call) is unaffected either way, since eviction only
    removes the registry's *reference*, it doesn't touch the Agent object
    itself; a concurrently in-flight request keeps working against the
    object it already has, it just won't find it back in the registry
    under its conversation_id afterwards (the next request for that id
    will get a freshly created Agent instead — see `get_agent`).

    `Agent.is_idle()` is what actually protects against evicting a *busy*
    Agent (see agent/agent.py's `_in_flight` counter) — this function
    doesn't need to (and doesn't) special-case in-flight requests itself.

    Returns the number of conversations evicted.
    """
    idle_timeout = settings.AGENT_IDLE_TIMEOUT_SECONDS

    with _lock:
        stale_ids = [
            cid for cid, agent in _agents.items() if agent.is_idle(idle_timeout)
        ]
        for cid in stale_ids:
            del _agents[cid]

    if stale_ids:
        log.info(
            f"Agent cleanup: evicted {len(stale_ids)} idle conversation(s) "
            f"(remaining in RAM: {len(_agents)})"
        )
    return len(stale_ids)


def _cleanup_loop() -> None:
    interval = settings.AGENT_CLEANUP_INTERVAL_SECONDS
    while True:
        time.sleep(interval)
        try:
            _cleanup_once()
        except Exception:
            log.exception("Agent cleanup pass failed")


# Started once at import time (this module is a process-wide singleton
# registry, same as `_agents`/`_lock` above) — daemon so it never blocks
# process shutdown.
_cleanup_thread = threading.Thread(target=_cleanup_loop, name="agent-cleanup", daemon=True)
_cleanup_thread.start()
