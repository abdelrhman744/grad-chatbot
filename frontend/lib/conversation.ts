// Per-tab conversation identity for the chat backend (see backend Issue 2
// investigation: every request used to hardcode conversation_id: "default",
// so every browser tab/user shared one Agent, one memory, one active
// document on the server).
//
// sessionStorage (not localStorage) is deliberate: it is scoped to a single
// browser tab, so two tabs never see the same value, while a reload of the
// SAME tab keeps reusing it (sessionStorage survives navigation/refresh and
// is only cleared when the tab closes) — exactly "persist for the lifetime
// of that browser tab" from the requirements.

const STORAGE_KEY = "chat_conversation_id";

/**
 * Returns this tab's conversation id, creating and persisting one on first
 * call. Must only be called on the client (guards `typeof window` so it's
 * safe to reference from a lazy useState initializer that also runs during
 * Next.js server rendering, where it returns "" — no chat request is ever
 * sent during SSR, so that placeholder is never actually used to talk to
 * the backend).
 */
export function getConversationId(): string {
  if (typeof window === "undefined") {
    return "";
  }

  const existing = window.sessionStorage.getItem(STORAGE_KEY);
  if (existing) {
    return existing;
  }

  const id = crypto.randomUUID();
  window.sessionStorage.setItem(STORAGE_KEY, id);
  return id;
}
