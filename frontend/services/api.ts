const BASE = "/api";

// Requests can legitimately take a little while (LLM generation + Qdrant
// retrieval), but they should never hang indefinitely. If the backend (or
// the Next.js dev-server proxy in front of it) stalls past this, we abort
// client-side and surface a clear timeout error instead of letting the
// browser's own generic network error message ("Failed to fetch" / a
// proxy's blank 500) reach the user unexplained.
const REQUEST_TIMEOUT_MS = 60_000;

export type Language = "auto" | "ar" | "en";

export interface ChatResponse {
  answer: string;
  sources: string;
  stt_text: string;
}

export interface UploadResponse {
  message: string;
  chunks_added: number;
  stored_files?: any[];
}

/**
 * Fetch wrapper that:
 *  - applies a client-side timeout (AbortController) so a stalled request
 *    fails fast with a clear message instead of hanging forever.
 *  - on a non-2xx response, tries to parse the FastAPI JSON error body
 *    ({"detail": "..."}) and falls back to a *descriptive* message that
 *    includes the HTTP status when the body isn't JSON (e.g. the request
 *    never reached FastAPI at all — a proxy/network failure upstream of
 *    the backend — in which case there is no `detail` field to read).
 */
async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { ...init, signal: controller.signal });
  } catch (err: any) {
    if (err?.name === "AbortError") {
      throw new Error(
        `Request timed out after ${REQUEST_TIMEOUT_MS / 1000}s. The backend may be ` +
          "unreachable or overloaded — check that the FastAPI server is running."
      );
    }
    // fetch() only throws for network-level failures (DNS, connection
    // refused, CORS, etc.) — i.e. the request never got an HTTP response
    // at all. Surface that distinctly from an HTTP error status.
    throw new Error(`Network error while calling ${path}: ${err?.message || err}`);
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
  language: Language = "auto"
): Promise<ChatResponse> {
  const res = await apiFetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, language }),
  });

  return res.json();
}

export async function askVoice(
  audioBlob: Blob,
  language: Language = "auto"
): Promise<ChatResponse> {
  const form = new FormData();
  form.append("audio", audioBlob, "recording.webm");
  form.append("language", language);

  const res = await apiFetch("/chat/voice", {
    method: "POST",
    body: form,
  });

  return res.json();
}

export async function resetConversation(): Promise<void> {
  await apiFetch("/chat/reset", { method: "POST" });
}

export async function uploadFiles(files: File[]): Promise<UploadResponse> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));

  const res = await apiFetch("/upload", {
    method: "POST",
    body: form,
  });

  return res.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}
