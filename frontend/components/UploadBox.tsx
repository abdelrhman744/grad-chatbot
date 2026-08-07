"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { UploadCloud, Download, FolderOpen, CheckCircle2, XCircle, RotateCcw } from "lucide-react";
import { uploadFiles, getStoredFiles, StoredFile, UploadStage } from "@/services/api";
import { getConversationId } from "@/lib/conversation";
import { getFileTypeMeta } from "@/lib/fileTypeMeta";
import Card from "@/components/ui/Card";
import Badge from "@/components/ui/Badge";
import Skeleton from "@/components/ui/Skeleton";
import EmptyState from "@/components/ui/EmptyState";

const ACCEPTED = ".pdf,.docx,.doc,.txt,.md,.json,.png,.jpg,.jpeg,.xlsx,.xls,.csv";

const STAGE_LABELS: Record<UploadStage, string> = {
  queued: "Uploading…",
  parsing: "Parsing document…",
  chunking: "Chunking content…",
  embedding: "Indexing for search…",
  done: "Finishing up…",
};

interface Props {
  onUploaded: (msg: string, chunks: number) => void;
}

function relativeTime(iso: string | null | undefined): string {
  if (!iso) return "";
  const then = new Date(iso.replace(" ", "T")).getTime();
  if (Number.isNaN(then)) return "";
  const diffMs = Date.now() - then;
  const mins = Math.floor(diffMs / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export default function UploadBox({ onUploaded }: Props) {
  // This tab's own conversation id (see lib/conversation.ts) — every file
  // uploaded here is tagged with it server-side and only ever retrievable
  // within this conversation (see Document Isolation).
  const [conversationId] = useState<string>(() => getConversationId());
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState<UploadStage | null>(null);
  const [filesLoading, setFilesLoading] = useState(true);
  const [status, setStatus] = useState<{ ok: boolean; msg: string } | null>(null);
  const [files, setFiles] = useState<StoredFile[]>([]);
  const [lastFiles, setLastFiles] = useState<File[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  const refreshFiles = useCallback(async () => {
    try {
      setFiles(await getStoredFiles());
    } catch {
      // Non-fatal — the sidebar just won't show the stored-files list.
    } finally {
      setFilesLoading(false);
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
      setStage("queued");
      setLastFiles(selected);
      try {
        const result = await uploadFiles(selected, conversationId, setStage);
        if (result.status === "error") {
          setStatus({ ok: false, msg: result.error || "Upload failed." });
        } else {
          const chunks = result.chunks_added ?? 0;
          const msg = `Processed ${selected.length} file(s) successfully (${chunks} chunks added)`;
          setStatus({ ok: true, msg });
          onUploaded(msg, chunks);
          await refreshFiles();
        }
      } catch (e: any) {
        setStatus({ ok: false, msg: e.message });
      } finally {
        setLoading(false);
        setStage(null);
      }
    },
    [onUploaded, refreshFiles, conversationId]
  );

  const retry = useCallback(() => {
    if (lastFiles.length) handleFiles(lastFiles);
  }, [handleFiles, lastFiles]);

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
    <div className="w-full flex flex-col gap-5">
      {/* Drop zone */}
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`
          relative cursor-pointer rounded-2xl border-2 border-dashed p-7
          flex flex-col items-center gap-3 text-center select-none
          transition-all duration-200
          ${dragging
            ? "border-primary bg-primary/5 scale-[1.01] shadow-glow"
            : "border-border hover:border-primary/40 hover:bg-surface-2/60"}
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

        <div className={`w-12 h-12 rounded-full flex items-center justify-center transition-colors
          ${dragging ? "bg-gradient-primary" : "bg-surface-2"}`}>
          <UploadCloud className={`w-5 h-5 ${dragging ? "text-white" : "text-muted"}`} strokeWidth={1.75} />
        </div>

        <div>
          <p className="font-medium text-ink text-sm">
            {dragging ? "Drop your files here" : "Upload documents"}
          </p>
          <p className="text-xs text-muted mt-1 leading-relaxed">
            PDF, Word, Excel, CSV, Text, JSON, Images
          </p>
        </div>

        {loading && (
          <div className="w-full mt-1">
            <div className="h-1.5 w-full rounded-full bg-surface-2 overflow-hidden">
              <div className="h-full w-1/3 rounded-full bg-gradient-primary animate-[upload-progress_1.1s_ease-in-out_infinite]" />
            </div>
            <p className="text-xs text-muted mt-2">
              {stage ? STAGE_LABELS[stage] : "Processing…"}
            </p>
          </div>
        )}
      </div>

      {/* Status */}
      {status && (
        <div className="-mt-2 flex items-center justify-between gap-2 animate-fade-in">
          <p
            className={`flex items-center gap-1.5 text-xs font-medium ${
              status.ok ? "text-success" : "text-danger"
            }`}
          >
            {status.ok ? <CheckCircle2 className="w-3.5 h-3.5 flex-shrink-0" /> : <XCircle className="w-3.5 h-3.5 flex-shrink-0" />}
            <span className="truncate">{status.msg}</span>
          </p>
          {!status.ok && !loading && lastFiles.length > 0 && (
            <button
              type="button"
              onClick={retry}
              className="flex items-center gap-1 text-xs font-medium text-muted hover:text-ink transition-colors flex-shrink-0"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Try again
            </button>
          )}
        </div>
      )}

      {/* Stored files */}
      <div>
        <p className="text-xs font-mono text-muted uppercase tracking-widest mb-2.5">
          Recent Documents
        </p>

        {filesLoading ? (
          <div className="space-y-2">
            <Skeleton className="h-12 w-full" />
            <Skeleton className="h-12 w-full" />
            <Skeleton className="h-12 w-full" />
          </div>
        ) : files.length === 0 ? (
          <EmptyState
            icon={FolderOpen}
            title="No documents yet"
            subtitle="Upload a file above to start building your knowledge base"
          />
        ) : (
          <ul className="space-y-2 max-h-[340px] overflow-y-auto pr-0.5">
            {files.map((f) => {
              const meta = getFileTypeMeta(f.file_type);
              const Icon = meta.icon;
              return (
                <li key={f.filename}>
                  <Card padding="sm" hoverable className="flex items-center gap-2.5">
                    <div className="w-8 h-8 rounded-lg bg-surface-2 flex items-center justify-center flex-shrink-0">
                      <Icon className="w-4 h-4 text-muted" />
                    </div>

                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-ink truncate" title={f.filename}>
                        {f.filename}
                      </p>
                      <div className="flex items-center gap-1.5 mt-0.5">
                        <Badge color={meta.color}>{meta.label}</Badge>
                        <span className="text-[11px] text-muted">
                          {f.chunks} chunks{f.processed_at ? ` · ${relativeTime(f.processed_at)}` : ""}
                        </span>
                      </div>
                    </div>

                    {f.download_url && (
                      <a
                        href={f.download_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        title="Download"
                        className="w-7 h-7 rounded-lg flex items-center justify-center text-muted
                          hover:text-ink hover:bg-surface-3 transition-all flex-shrink-0"
                      >
                        <Download className="w-3.5 h-3.5" />
                      </a>
                    )}
                  </Card>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}
