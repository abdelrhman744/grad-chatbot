"use client";

import { useEffect, useState } from "react";
import { Sparkles, RotateCcw, PanelLeftClose, PanelLeft, X, ScanText } from "lucide-react";
import ChatBox from "@/components/ChatBox";
import UploadBox from "@/components/UploadBox";
import HandwrittenOcrModal from "@/components/HandwrittenOcrModal";
import Card from "@/components/ui/Card";
import { checkHealth } from "@/services/api";

export default function Home() {
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [uploadNote, setUploadNote] = useState<string>("");
  const [resetSignal, setResetSignal] = useState(0);
  const [ocrOpen, setOcrOpen] = useState(false);

  useEffect(() => {
    checkHealth().then(setHealthy);
  }, []);

  return (
    <div className="flex h-screen overflow-hidden bg-base" dir="ltr">
      {/* Mobile backdrop, shown behind the drawer when open on small screens */}
      {sidebarOpen && (
        <div
          onClick={() => setSidebarOpen(false)}
          className="fixed inset-0 z-30 bg-black/60 backdrop-blur-sm md:hidden animate-fade-in"
        />
      )}

      <aside
        className={`
          fixed inset-y-0 left-0 z-40 md:relative md:z-auto
          flex flex-col border-r border-border glass md:bg-surface md:backdrop-blur-none
          transition-all duration-300 ease-in-out flex-shrink-0
          ${sidebarOpen ? "w-[19rem] translate-x-0" : "w-[19rem] -translate-x-full md:w-0 md:translate-x-0 md:overflow-hidden"}
        `}
      >
        <div className="flex flex-col h-full p-5 gap-5 min-w-[19rem]">
          <div className="flex items-center justify-between pt-1">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-gradient-primary flex items-center justify-center flex-shrink-0 shadow-glow">
                <Sparkles className="w-4 h-4 text-white" />
              </div>
              <span className="font-semibold text-ink tracking-tight">DocAssist AI</span>
            </div>
            <button
              onClick={() => setSidebarOpen(false)}
              className="md:hidden p-1.5 rounded-lg text-muted hover:text-ink hover:bg-surface-2 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="flex items-center gap-2 text-xs px-0.5">
            <span
              className={`w-2 h-2 rounded-full ${
                healthy === null
                  ? "bg-muted animate-pulse"
                  : healthy
                  ? "bg-success shadow-[0_0_8px_rgba(16,185,129,0.6)]"
                  : "bg-danger shadow-[0_0_8px_rgba(239,68,68,0.6)]"
              }`}
            />
            <span className="text-muted">
              {healthy === null
                ? "Connecting…"
                : healthy
                ? "AI Assistant online"
                : "AI Assistant offline"}
            </span>
          </div>

          <Card padding="md" className="flex-1 min-h-0 flex flex-col overflow-hidden">
            <p className="text-xs font-mono text-muted uppercase tracking-widest mb-3 flex-shrink-0">
              Knowledge Base
            </p>

            <div className="flex-1 min-h-0 overflow-y-auto -mx-1 px-1">
              <UploadBox
                onUploaded={(msg, chunks) =>
                  setUploadNote(`${chunks} chunks added`)
                }
              />
            </div>

            {uploadNote && (
              <p className="mt-2 text-xs text-success font-medium flex-shrink-0 animate-fade-in">
                {uploadNote}
              </p>
            )}
          </Card>

          <button
            onClick={() => setResetSignal((n) => n + 1)}
            className="w-full h-10 px-3 rounded-xl flex items-center justify-center gap-2
              text-sm font-medium text-muted bg-surface-2 hover:text-ink hover:bg-surface-3
              border border-transparent hover:border-border-light
              transition-all flex-shrink-0"
            title="Clear conversation memory"
          >
            <RotateCcw className="w-4 h-4" />
            New Conversation
          </button>

          <div className="pt-4 border-t border-border flex-shrink-0">
            <p className="text-xs text-muted leading-relaxed">
              Powered by <strong className="text-ink">AI Assistant</strong> ·{" "}
              <strong className="text-ink">Knowledge Base</strong> ·{" "}
              <strong className="text-ink">Smart Chat</strong>
            </p>
            <p className="text-xs text-subtle mt-1">
              Intelligent Search · Arabic &amp; English support
            </p>
          </div>
        </div>
      </aside>

      <main className="flex flex-col flex-1 min-w-0 h-full">
        <header className="flex items-center gap-3 px-4 py-3 border-b border-border bg-surface/80 backdrop-blur-md flex-shrink-0">
          <button
            onClick={() => setSidebarOpen((o) => !o)}
            className="p-1.5 rounded-lg text-muted hover:text-ink hover:bg-surface-2 transition-colors"
            title="Toggle sidebar"
          >
            {sidebarOpen ? (
              <PanelLeftClose className="w-5 h-5" strokeWidth={1.75} />
            ) : (
              <PanelLeft className="w-5 h-5" strokeWidth={1.75} />
            )}
          </button>

          <h1 className="text-sm font-semibold text-ink">Document Chat</h1>

          <button
            onClick={() => setOcrOpen(true)}
            className="ml-auto flex items-center gap-1.5 h-8 px-3 rounded-lg
              text-xs font-medium text-muted bg-surface-2 hover:text-ink hover:bg-surface-3
              border border-transparent hover:border-border-light transition-all"
            title="Extract text from a handwritten image (Arabic or English)"
          >
            <ScanText className="w-3.5 h-3.5" />
            Handwritten OCR
          </button>

          <div className="text-xs text-subtle font-mono hidden sm:block">
            Document Analysis · Intelligent Search
          </div>
        </header>

        <div className="flex-1 min-h-0">
          <ChatBox resetSignal={resetSignal} />
        </div>
      </main>

      <HandwrittenOcrModal open={ocrOpen} onClose={() => setOcrOpen(false)} />
    </div>
  );
}
