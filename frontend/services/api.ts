const BASE = "/api";

// Requests can legitimately take a little while (LLM generation + Qdrant
// retrieval), but they should never hang indefinitely. If the backend (or
// the Next.js dev-server proxy in front of it) stalls past this, we abort
// client-side and surface a clear timeout error instead of letting the
// browser's own generic network error message ("Failed to fetch" / a
// proxy's blank 500) reach the user unexplained.
const REQUEST_TIMEOUT_MS = 60_000;

export type Language = "auto" | "ar" | "en";

export interface ReportInfo {
  filename: string;
  object_name: string;
  download_url: string | null;
  proxy_download_path: string;
}

export interface ChatResponse {
  answer: string;
  sources: string;
  stt_text: string;
  report?: ReportInfo | null;
}

export interface StoredFile {
  filename: string;
  stored_path: string;
  file_type: string;
  chunks: number;
  processed_at: string;
  download_url: string | null;
}

export type UploadStage = "queued" | "parsing" | "chunking" | "embedding" | "done";

export interface UploadJobStatus {
  job_id: string;
  status: "queued" | "processing" | "done" | "error";
  stage: UploadStage;
  chunks_added?: number;
  stored_files?: StoredFile[];
  error?: string;
}

export interface ReportResponse {
  message: string;
  object_name: string;
  download_url: string | null;
  proxy_download_path: string;
  language: "ar" | "en";
}

/**
<<<<<<< HEAD
=======
 * Thrown only when fetch() itself never got an HTTP response at all — DNS
 * failure, connection refused, or (notably on Windows/Docker Desktop's
 * WSL2 networking) a transient net::ERR_CONNECTION_RESET on the
 * host<->container port-forward. This is distinct from a real HTTP error
 * status (server responded, just not with 2xx) and from the AbortController
 * timeout below — both of those indicate an actual problem worth surfacing
 * immediately, whereas this one is often gone on the very next attempt.
 */
class NetworkError extends Error {
  constructor(path: string, cause: unknown) {
    super(`Network error while calling ${path}: ${(cause as any)?.message || cause}`);
    this.name = "NetworkError";
  }
}

/**
>>>>>>> aac0288 (Initial commit - LASTVERSION)
 * Fetch wrapper that:
 *  - applies a client-side timeout (AbortController) so a stalled request
 *    fails fast with a clear message instead of hanging forever.
 *  - on a non-2xx response, tries to parse the FastAPI JSON error body
 *    ({"detail": "..."}) and falls back to a *descriptive* message that
 *    includes the HTTP status when the body isn't JSON (e.g. the request
 *    never reached FastAPI at all — a proxy/network failure upstream of
 *    the backend — in which case there is no `detail` field to read).
 */
async function apiFetch(
  path: string,
  init: RequestInit = {},
  timeoutMs: number = REQUEST_TIMEOUT_MS
): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { ...init, signal: controller.signal });
  } catch (err: any) {
    if (err?.name === "AbortError") {
      throw new Error(
        `Request timed out after ${timeoutMs / 1000}s. The backend may be ` +
          "unreachable or overloaded — check that the FastAPI server is running."
      );
    }
    // fetch() only throws for network-level failures (DNS, connection
    // refused, CORS, etc.) — i.e. the request never got an HTTP response
    // at all. Surface that distinctly from an HTTP error status.
<<<<<<< HEAD
    throw new Error(`Network error while calling ${path}: ${err?.message || err}`);
=======
    throw new NetworkError(path, err);
>>>>>>> aac0288 (Initial commit - LASTVERSION)
  } finally {
    clearTimeout(timeout);
  }

  if (!res.ok) {
    let detail: string | undefined;
    try {
      const body = await res.json();
      detail = body?.detail;
    } catch {
      // Response body wasn't JSON. This happens when the error did NOT
      // come from our FastAPI handlers (which always return
      // {"detail": "..."}) — e.g. a proxy/timeout/connection error
      // upstream of the backend. Report the real HTTP status instead of
      // a misleading generic message.
      detail = undefined;
    }
    throw new Error(
      detail || `Request to ${path} failed with HTTP ${res.status} (${res.statusText || "no response body"})`
    );
  }

  return res;
}

export async function askQuestion(
  query: string,
  language: Language = "auto",
  conversationId: string
): Promise<ChatResponse> {
  const res = await apiFetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, language, conversation_id: conversationId }),
  });

  return res.json();
}

export async function askVoice(
  audioBlob: Blob,
  language: Language = "auto",
  conversationId: string
): Promise<ChatResponse> {
  const form = new FormData();
  form.append("audio", audioBlob, "recording.webm");
  form.append("language", language);
  form.append("conversation_id", conversationId);

  const res = await apiFetch("/chat/voice", {
    method: "POST",
    body: form,
  });

  return res.json();
}

export async function resetConversation(conversationId: string): Promise<void> {
  await apiFetch(`/chat/reset?conversation_id=${encodeURIComponent(conversationId)}`, {
    method: "POST",
  });
}

const UPLOAD_POLL_INTERVAL_MS = 1000;
// Generous cap so even a very large/slow ingestion isn't abandoned
// prematurely — this only bounds the polling loop, not the backend job
// itself (which keeps running regardless).
const UPLOAD_POLL_TIMEOUT_MS = 15 * 60_000;

async function startUploadJob(files: File[], conversationId: string): Promise<{ job_id: string }> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  // Required — every uploaded document is tagged with this conversation_id
  // server-side and only ever retrievable within it (see Document
  // Isolation). No fallback: an upload with no owning conversation_id
  // would either be rejected or, worse, silently land in the wrong scope.
  form.append("conversation_id", conversationId);

  const res = await apiFetch("/upload", {
    method: "POST",
    body: form,
  });

  return res.json();
}

<<<<<<< HEAD
async function getUploadJobStatus(jobId: string): Promise<UploadJobStatus> {
  const res = await apiFetch(`/upload/status/${jobId}`);
  return res.json();
=======
// A single poll occasionally fails with a transient network-layer reset
// before it ever reaches the backend — most commonly net::ERR_CONNECTION_RESET
// on Windows/Docker Desktop's WSL2 host<->container port-forward — even
// though the backend job is running fine and the very next poll a moment
// later succeeds. Retry just that case a few times within this single poll
// iteration (not bypassing uploadFiles()'s overall UPLOAD_POLL_TIMEOUT_MS
// deadline) before letting it propagate as a real error. A real HTTP error
// status or the AbortController timeout is NOT retried here — those mean
// the request did get through and something is actually wrong.
const STATUS_POLL_RETRY_DELAYS_MS = [500, 1000, 1500];

async function getUploadJobStatus(jobId: string): Promise<UploadJobStatus> {
  for (let attempt = 0; ; attempt++) {
    try {
      const res = await apiFetch(`/upload/status/${jobId}`);
      return res.json();
    } catch (err) {
      const isLastAttempt = attempt >= STATUS_POLL_RETRY_DELAYS_MS.length;
      if (!(err instanceof NetworkError) || isLastAttempt) {
        throw err;
      }
      await new Promise((resolve) =>
        setTimeout(resolve, STATUS_POLL_RETRY_DELAYS_MS[attempt])
      );
    }
  }
>>>>>>> aac0288 (Initial commit - LASTVERSION)
}

/**
 * Starts an upload job (returns almost immediately — the heavy parse ->
 * chunk -> embed -> index pipeline runs in the background) then polls its
 * status until it reaches "done" or "error", invoking `onStage` on every
 * stage change so the caller can show real progress instead of a single
 * opaque spinner for the whole ingestion duration.
 */
export async function uploadFiles(
  files: File[],
  conversationId: string,
  onStage?: (stage: UploadStage) => void
): Promise<UploadJobStatus> {
  const { job_id } = await startUploadJob(files, conversationId);

  const deadline = Date.now() + UPLOAD_POLL_TIMEOUT_MS;
  let lastStage: UploadStage | null = null;

  while (true) {
    const status = await getUploadJobStatus(job_id);

    if (status.stage !== lastStage) {
      lastStage = status.stage;
      onStage?.(status.stage);
    }

    if (status.status === "done" || status.status === "error") {
      return status;
    }

    if (Date.now() > deadline) {
      throw new Error(
        "This upload is taking longer than expected. It's likely still " +
          "processing in the background — check the document list again in a bit."
      );
    }

    await new Promise((resolve) => setTimeout(resolve, UPLOAD_POLL_INTERVAL_MS));
  }
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

export async function getStoredFiles(): Promise<StoredFile[]> {
  const res = await apiFetch("/stored-files");
  const body = await res.json();
  return body.files ?? [];
}

// ── Handwritten OCR (local TrOCR, free — see backend/routes/ocr.py) ────────

export type HandwrittenLanguage = "ar" | "en";

export interface HandwrittenOcrResult {
  text: string;
  language: HandwrittenLanguage;
  type: "handwritten";
  indexed?: boolean;
  chunks_added?: number;
}

// The very first request for a given language triggers a one-time model
// download from Hugging Face on the backend (can take a couple of minutes
// on a slow connection) — a much longer client-side timeout than the
// default 60s is used so that cold start doesn't surface as a false
// "request timed out" error. Warm requests return in a few seconds.
const OCR_REQUEST_TIMEOUT_MS = 5 * 60_000;

export async function ocrHandwritten(
  file: File,
  language: HandwrittenLanguage
): Promise<HandwrittenOcrResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("language", language);

  const res = await apiFetch(
    "/ocr/handwritten",
    { method: "POST", body: form },
    OCR_REQUEST_TIMEOUT_MS
  );
  return res.json();
}

export async function generateReport(filename: string): Promise<ReportResponse> {
  const res = await apiFetch("/reports/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename }),
  });
  return res.json();
}

// ── Streaming chat over WebSocket ───────────────────────────────────────────
//
// Backend protocol (backend/routes/ws.py):
//   send    -> { query, language, conversation_id }
//   receive -> { type: "start" }
//            | { type: "status", text }         (e.g. "Generating the report...")
//            | { type: "token", text }          (repeated, in order)
//            | { type: "done", answer, sources, report? }
//            | { type: "error", message }

function wsUrl(): string {
  const configured = process.env.NEXT_PUBLIC_WS_URL;
  if (configured) return configured;
  if (typeof window === "undefined") return "ws://localhost:8000/ws/chat";
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  // Dev setup: Next.js runs on its own port, FastAPI is on :8000.
  const host = window.location.hostname;
  return `${proto}//${host}:8000/ws/chat`;
}

export interface StreamChatHandlers {
  onStatus?: (text: string) => void;
  onToken: (text: string) => void;
  onDone: (result: { answer: string; sources: string; report?: ReportInfo | null }) => void;
  onError: (message: string) => void;
}

/**
 * Opens a WebSocket, sends one question, streams the answer token by
 * token via `onToken`, then resolves once `onDone`/`onError` fires and
 * the socket is closed. Each call opens a fresh connection — simplest
 * and most robust option for a single request/response turn.
 */
export function streamChat(
  query: string,
  language: Language,
  conversationId: string,
  handlers: StreamChatHandlers
): () => void {
  const socket = new WebSocket(wsUrl());
  let settled = false;

  const fail = (message: string) => {
    if (settled) return;
    settled = true;
    handlers.onError(message);
    socket.close();
  };

  socket.onopen = () => {
    // No fallback to a shared "default" id: conversationId is this tab's
    // own identity (see lib/conversation.ts) and must always be real —
    // silently defaulting here is exactly what let unrelated conversations
    // merge (see Issue 2 investigation).
    socket.send(JSON.stringify({ query, language, conversation_id: conversationId }));
  };

  socket.onmessage = (event) => {
    let msg: any;
    try {
      msg = JSON.parse(event.data);
    } catch {
      return;
    }

    if (msg.type === "token") {
      handlers.onToken(msg.text as string);
    } else if (msg.type === "status") {
      handlers.onStatus?.(msg.text as string);
    } else if (msg.type === "done") {
      settled = true;
      handlers.onDone({ answer: msg.answer ?? "", sources: msg.sources ?? "", report: msg.report ?? null });
      socket.close();
    } else if (msg.type === "error") {
      fail(msg.message || "Streaming error");
    }
  };

  socket.onerror = () => fail("WebSocket connection error.");
  socket.onclose = () => {
    if (!settled) fail("Connection closed before a response was received.");
  };

  // Returns a cancel function the caller can use to abort mid-stream.
  return () => {
    settled = true;
    socket.close();
  };
<<<<<<< HEAD
}
=======
}
>>>>>>> aac0288 (Initial commit - LASTVERSION)
