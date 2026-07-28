"""
session.py

Keeps one Agent instance per conversation_id alive across calls, so
short-term memory (and the loaded long-term summary) persists between
turns without re-reading disk every time. This is an in-process
registry; it resets if the process restarts (the persisted long-term
summary on disk survives restarts regardless).
"""

from __future__ import annotations

import threading

from .agent import Agent

_agents: dict[str, Agent] = {}
_lock = threading.Lock()

DEFAULT_CONVERSATION_ID = "default"


def get_agent(conversation_id: str = DEFAULT_CONVERSATION_ID) -> Agent:
    with _lock:
        if conversation_id not in _agents:
            _agents[conversation_id] = Agent(conversation_id=conversation_id)
        return _agents[conversation_id]


def reset_agent(conversation_id: str = DEFAULT_CONVERSATION_ID) -> None:
    with _lock:
        agent = _agents.get(conversation_id)
        if agent:
            agent.reset_memory()
