"""
session.py

Keeps one Agent instance per conversation_id alive across requests, so
short-term memory (and the loaded long-term summary) persists between
turns without re-reading disk every time. This is an in-process registry;
it resets when the backend restarts (the persisted long-term summary on
disk survives restarts regardless).
"""

from __future__ import annotations

import threading

from config import settings

from .agent import Agent

_agents: dict[str, Agent] = {}
_lock = threading.Lock()


def get_agent(conversation_id: str = settings.DEFAULT_CONVERSATION_ID) -> Agent:
    with _lock:
        if conversation_id not in _agents:
            _agents[conversation_id] = Agent(conversation_id=conversation_id)
        return _agents[conversation_id]


def reset_agent(conversation_id: str = settings.DEFAULT_CONVERSATION_ID) -> None:
    with _lock:
        agent = _agents.get(conversation_id)
        if agent:
            agent.reset_memory()
