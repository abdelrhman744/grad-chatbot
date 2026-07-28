"use client";

import { useCallback, useRef, useState } from "react";
import { uploadFiles, uploadXlsx } from "@/services/api";

interface Props {
  onUploaded: (msg: string, chunks: number) => void;
}

type Mode = "docs" | "xlsx";

export default function UploadBox({ onUploaded }: Props) {
  const [mode, setMode] = useState<Mode>("docs");
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<{ ok: boolean; msg: string } | null>(null);
  const [fileNames, setFileNames] = useState<string[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  const accept =
    mode === "docs"
      ? ".pdf,.docx,.doc,.txt,.png,.jpg,.jpeg,.tiff,.bmp,.webp,.json"
      : ".xlsx,.xls,.json";

  const hint =
    mode === "docs"
      ? "PDF / image / txt + matching metadata .json (same base name)"
      : "Excel (.xlsx) + matching metadata .json (same base name)";

  const handleFiles = useCallback(
    async (files: File[]) => {
      if (!files.length) return;

      const stems = new Map<string, { doc?: string; json?: string }>();
      for (const f of files) {
        const base = f.name.replace(/\.[^.]+$/, "");
        const entry = stems.get(base) || {};
        if (f.name.toLowerCase().endsWith(".json")) entry.json = f.name;
        else entry.doc = f.name;
        stems.set(base, entry);
      }
      const unpaired = [...stems.entries()].filter(
        ([, v]) => (v.doc && !v.json) || (v.json && !v.doc)
      );
      if (unpaired.length) {
        setStatus({
          ok: false,
          msg:
            "Each file must be uploaded with its metadata JSON (same base name). Missing pairs for: " +
            unpaired.map(([k]) => k).join(", "),
        });
        setFileNames(files.map((f) => f.name));
        return;
      }

      setFileNames(files.map((f) => f.name));
      setLoading(true);
      setStatus(null);
      try {
        const res =
          mode === "docs" ? await uploadFiles(files) : await uploadXlsx(files);
        setStatus({
          ok: true,
          msg: `${res.message} (${res.chunks_added} chunks added)`,
        });
        onUploaded(res.message, res.chunks_added);
      } catch (e: any) {
        setStatus({ ok: false, msg: e.message });
      } finally {
        setLoading(false);
      }
    },
    [onUploaded, mode]
  );

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    handleFiles(Array.from(e.dataTransfer.files));
  };

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFiles(Array.from(e.target.files ?? []));
    e.target.value = "";
  };

  return (
    <div className="w-full">
      <div className="flex gap-1 mb-3 p-1 rounded-lg bg-ash/60">
        <button
          type="button"
          onClick={() => {
            setMode("docs");
            setStatus(null);
            setFileNames([]);
          }}
          className={`flex-1 h-8 text-xs font-medium rounded-md transition-all ${
            mode === "docs"
              ? "bg-white text-ink shadow-sm"
              : "text-muted hover:text-ink"
          }`}
        >
          PDF / Docs
        </button>
        <button
          type="button"
          onClick={() => {
            setMode("xlsx");
            setStatus(null);
            setFileNames([]);
          }}
          className={`flex-1 h-8 text-xs font-medium rounded-md transition-all ${
            mode === "xlsx"
              ? "bg-white text-ink shadow-sm"
              : "text-muted hover:text-ink"
          }`}
        >
          Excel
        </button>
      </div>

      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`
          relative cursor-pointer rounded-xl border-2 border-dashed p-6
          flex flex-col items-center gap-3 text-center select-none
          transition-all duration-200
          ${
            dragging
              ? "border-teal bg-teal/5 scale-[1.01]"
              : "border-ash hover:border-muted hover:bg-ash/40"
          }
        `}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={accept}
          className="hidden"
          onChange={onInputChange}
        />

        <div
          className={`w-10 h-10 rounded-full flex items-center justify-center ${
            dragging ? "bg-teal/20" : "bg-ash"
          }`}
        >
          <svg
            className={`w-5 h-5 ${dragging ? "text-teal" : "text-muted"}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1.5}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
            />
          </svg>
        </div>

        <div>
          <p className="font-medium text-ink text-sm">
            {dragging
              ? "Drop your files here"
              : mode === "docs"
              ? "Upload documents"
              : "Upload spreadsheets"}
          </p>
          <p className="text-xs text-muted mt-1 leading-relaxed">{hint}</p>
        </div>

        {loading && (
          <div className="flex items-center gap-2 text-sm text-muted">
            <svg
              className="w-4 h-4 animate-spin-slow text-teal"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
              />
            </svg>
            Processing…
          </div>
        )}
      </div>

      {fileNames.length > 0 && !loading && (
        <ul className="mt-3 space-y-1">
          {fileNames.map((n) => (
            <li key={n} className="flex items-center gap-2 text-sm text-muted">
              <span className="w-1.5 h-1.5 rounded-full bg-teal flex-shrink-0" />
              {n}
            </li>
          ))}
        </ul>
      )}

      {status && (
        <p
          className={`mt-3 text-sm font-medium ${
            status.ok ? "text-teal" : "text-accent"
          }`}
        >
          {status.ok ? "✓ " : "✗ "}
          {status.msg}
        </p>
      )}
    </div>
  );
}
