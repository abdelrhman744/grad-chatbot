"""
conversation_auth.py

conversation_id is a client-generated identifier used to scope a user's
documents, memory, and chat history (see frontend/lib/conversation.ts).
On its own it is a bare capability: whoever holds the string gets full
read/write access to that conversation, forever, and the server has no
way to tell "the browser tab that created this" apart from "anyone who
obtained this string by any means" — a copy-pasted URL, a proxy/access
log line, a shared screenshot, or simply crafting a request by hand
against a conversation_id seen somewhere.

This closes that gap the simplest way that fits the app's current
architecture (there is no user-account/login system to build a real
ownership model on top of): the SERVER, never the client, is the only
party that ever mints a conversation_id (see POST /api/session in
routes/chat.py), and it hands out an HMAC-SHA256 signature alongside it.
Every request that reads/writes data scoped by a conversation_id must
present that signature; the server recomputes and compares it
(constant-time) before touching anything. Forging a valid signature for
an arbitrary or guessed conversation_id requires settings.APP_SECRET_KEY,
which never leaves the server, so knowing (or successfully guessing) a
bare conversation_id string is no longer sufficient.

This is bearer-token-style protection (the same trust model as a signed
cookie or a single-claim JWT) — "whoever presents both the id and its
matching signature" is treated as the legitimate owner. It is the
"simplest viable fix" tier: it stops the realistic threats given this
app's current design (a leaked/guessed/hand-crafted bare conversation_id,
copy-pasted URLs, IDOR-style tampering with a request's conversation_id
field) without requiring a login system. It is not a substitute for real
user authentication if this app ever needs one.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

from fastapi import HTTPException

from config import settings


def new_conversation_id() -> str:
    """
    Generate a fresh, cryptographically random conversation id.

    Only ever called server-side, from POST /api/session — a client can
    never choose its own conversation_id. That's what makes the signature
    below meaningful: since the server is the sole source of ids, it only
    ever signs one it just generated itself, never one an attacker
    supplied hoping to get it signed.
    """
    return secrets.token_urlsafe(24)


def sign_conversation_id(conversation_id: str) -> str:
    """Deterministic HMAC-SHA256 signature for a conversation_id, keyed by
    the server-only settings.APP_SECRET_KEY."""
    return hmac.new(
        settings.APP_SECRET_KEY.encode("utf-8"),
        conversation_id.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def require_valid_conversation_token(conversation_id: str, token: str | None) -> None:
    """
    Raise HTTPException(403) unless `token` is a valid signature for
    `conversation_id`, previously minted by this server's own
    POST /api/session.

    Call this at the very top of every route that reads, writes, or
    deletes anything scoped by conversation_id, before touching any
    document/memory/agent state.
    """
    if not conversation_id or not token or not hmac.compare_digest(
        sign_conversation_id(conversation_id), token
    ):
        raise HTTPException(
            status_code=403,
            detail=(
                "Invalid or missing conversation token. Call POST /api/session "
                "to obtain a conversation_id and its token."
            ),
        )
