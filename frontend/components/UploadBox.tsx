"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { FileText } from "lucide-react";
import { uploadFiles, getStoredFiles, StoredFile } from "@/services/api";

const ACCEPTED = ".pdf,.docx,.doc,.txt,.md,.json,.png,.jpg,.jpeg";

interface Props {
  onUploaded: (msg: string, chunks: number) => void;
}

export default function UploadBox({ onUploaded }: Props) {
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<{ ok: boolean; msg: string } | null>(null);
  const [files, setFiles] = useState<StoredFile[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  const refreshFiles = useCallback(async () => {
    try {
      setFiles(await getStoredFiles());
    } catch {
      // Non-fatal — the sidebar just won't show the stored-files list.
    }
  }, []);

  useEffect(() => {
    refreshFiles();
  }, [refreshFiles]);

  const handleFiles = useCallback(
    async (selected: File[]) => {
      if (!selected.length) return;
      setLoading(true);
      setStatus(null);
      try {
        const res = await uploadFiles(selected);
        setStatus({ ok: true, msg: `${res.message} (${res.chunks_added} chunks added)` });
        onUploaded(res.message, res.chunks_added);
        await refreshFiles();
      } catch (e: any) {
        setStatus({ ok: false, msg: e.message });
      } finally {
        setLoading(false);
      }
    },
    [onUploaded, refreshFiles]
  );

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    handleFiles(Array.from(e.dataTransfer.files));
  };

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFiles(Array.from(e.target.files ?? []));
  };

  return (
    <div className="w-full">
      {/* Drop zone */}
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`
          relative cursor-pointer rounded-xl border-2 border-dashed p-8
          flex flex-col items-center gap-3 text-center select-none
          transition-all duration-200
          ${dragging
            ? "border-teal bg-teal/5 scale-[1.01]"
            : "border-ash hover:border-muted hover:bg-ash/40"}
        `}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPTED}
          className="hidden"
          onChange={onInputChange}
        />

        {/* Icon */}
        <div className={`w-12 h-12 rounded-full flex items-center justify-center
          ${dragging ? "bg-teal/20" : "bg-ash"}`}>
          <svg className={`w-6 h-6 ${dragging ? "text-teal" : "text-muted"}`}
            fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
          </svg>
        </div>

        <div>
          <p className="font-medium text-ink">
            {dragging ? "Drop your files here" : "Upload documents"}
          </p>
          <p className="text-sm text-muted mt-1">
            PDF, DOCX, TXT, JSON, Images — drag & drop or click
          </p>
        </div>

        {loading && (
          <div className="flex items-center gap-2 text-sm text-muted">
            <svg className="w-4 h-4 animate-spin-slow text-teal" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
            </svg>
            Processing…
          </div>
        )}
      </div>

      {/* Status */}
      {status && (
        <p className={`mt-3 text-sm font-medium ${status.ok ? "text-teal" : "text-accent"}`}>
          {status.ok ? "✓ " : "✗ "}{status.msg}
        </p>
      )}

      {/* Stored files — informational only. Ask the chatbot for a report
          ("generate a report", "اعمل تقرير") instead of a manual button;
          the agent resolves which uploaded file to use on its own. */}
      {files.length > 0 && (
        <ul className="mt-4 space-y-1.5">
          {files.map((f) => (
            <li key={f.filename} className="flex items-center gap-2 text-sm text-muted">
              <FileText className="w-3.5 h-3.5 flex-shrink-0 text-teal" />
              <span className="truncate" title={f.filename}>{f.filename}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
