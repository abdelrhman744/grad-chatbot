"use client";

import { useCallback, useRef, useState } from "react";
import { Loader2, Mic, Square } from "lucide-react";

interface Props {
  language: string;
  onResult: (blob: Blob) => Promise<void>;
  disabled?: boolean;
}

type State = "idle" | "recording" | "processing";

export default function VoiceRecorder({ onResult, disabled }: Props) {
  const [state, setState] = useState<State>("idle");
  const [seconds, setSeconds] = useState(0);

  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const cleanup = () => {
    mediaRef.current = null;
    chunksRef.current = [];
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;
    setSeconds(0);
  };

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const mr = new MediaRecorder(stream);
      chunksRef.current = [];

      mr.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mr.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });

        stream.getTracks().forEach((t) => t.stop());
        setState("processing");

        try {
          await onResult(blob);
        } catch (err) {
          console.error("Voice error:", err);
        } finally {
          cleanup();
          setState("idle");
        }
      };

      mr.start(100);
      mediaRef.current = mr;

      setSeconds(0);
      setState("recording");

      timerRef.current = setInterval(() => {
        setSeconds((s) => s + 1);
      }, 1000);
    } catch {
      alert("Microphone access denied.");
    }
  }, [onResult]);

  const stop = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    mediaRef.current?.stop();
  }, []);

  const fmt = (s: number) =>
    `${Math.floor(s / 60).toString().padStart(2, "0")}:${(s % 60)
      .toString()
      .padStart(2, "0")}`;

  return (
    <div className="flex items-center gap-2">
      {state === "idle" && (
        <button
          type="button"
          onClick={start}
          disabled={disabled}
          className="h-10 px-3 rounded-xl border border-ash bg-white text-ink
            flex items-center gap-2 text-sm font-medium
            hover:bg-ash/70 disabled:opacity-40 disabled:cursor-not-allowed
            transition-all flex-shrink-0"
        >
          <Mic className="w-4 h-4" />
          <span className="hidden sm:inline">Record</span>
        </button>
      )}

      {state === "recording" && (
        <button
          type="button"
          onClick={stop}
          className="h-10 px-3 rounded-xl bg-accent text-white
            flex items-center gap-2 text-sm font-medium
            hover:bg-accent/90 transition-all flex-shrink-0"
        >
          <Square className="w-4 h-4" />
          <span>Stop {fmt(seconds)}</span>
        </button>
      )}

      {state === "processing" && (
        <div
          className="h-10 px-3 rounded-xl border border-ash bg-white text-muted
          flex items-center gap-2 text-sm font-medium flex-shrink-0"
        >
          <Loader2 className="w-4 h-4 animate-spin" />
          <span className="hidden sm:inline">Transcribing</span>
        </div>
      )}
    </div>
  );
}