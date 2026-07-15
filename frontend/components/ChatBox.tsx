"use client";

import { useEffect, useRef, useState } from "react";
import { Globe2, Languages, Loader2, SendHorizontal, Mic2 } from "lucide-react";
import { askQuestion, askVoice, resetConversation, ChatResponse } from "@/services/api";
import AnswerBox from "./AnswerBox";
import SourceBox from "./SourceBox";
import VoiceRecorder from "./VoiceRecorder";

type Language = "auto" | "ar" | "en";

interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  sources?: string;
  stt?: string;
  loading?: boolean;
}

interface Props {
  resetSignal?: number;
}

const isArabicText = (text: string) => /[\u0600-\u06FF]/.test(text);

export default function ChatBox({ resetSignal }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [query, setQuery] = useState("");
  const [language, setLanguage] = useState<Language>("auto");
  const [submitting, setSubmitting] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isFirstRender = useRef(true);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;
    }
    resetConversation().catch(() => {});
    setMessages([]);
  }, [resetSignal]);

  const addUserMessage = (text: string, stt?: string) => {
    const id = crypto.randomUUID();

    setMessages((prev) => [
      ...prev,
      { id, role: "user", text, stt },
      { id: id + "-ai", role: "assistant", text: "", loading: true },
    ]);

    return id + "-ai";
  };

  const resolveAIMessage = (aiId: string, res: ChatResponse) => {
    setMessages((prev) =>
      prev.map((m) =>
        m.id === aiId
          ? {
              ...m,
              text: res.answer,
              sources: res.sources,
              loading: false,
            }
          : m
      )
    );
  };

  const failAIMessage = (aiId: string, err: string) => {
    setMessages((prev) =>
      prev.map((m) =>
        m.id === aiId
          ? {
              ...m,
              text: `Error: ${err}`,
              loading: false,
            }
          : m
      )
    );
  };

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();

    const q = query.trim();
    if (!q || submitting) return;

    setQuery("");
    setSubmitting(true);

    const aiId = addUserMessage(q);

    try {
      const res = await askQuestion(q, language);
      resolveAIMessage(aiId, res);
    } catch (err: any) {
      failAIMessage(aiId, err.message || "Request failed");
    } finally {
      setSubmitting(false);
    }
  };

  const handleVoice = async (blob: Blob) => {
    if (submitting) return;

    setSubmitting(true);

    const aiId = addUserMessage("Voice message…");

    try {
      const res = await askVoice(blob, language);

      setMessages((prev) =>
        prev.map((m) =>
          m.id === aiId.replace("-ai", "")
            ? { ...m, text: res.stt_text || "Voice message" }
            : m
        )
      );

      resolveAIMessage(aiId, res);
    } catch (err: any) {
      failAIMessage(aiId, err.message || "Voice processing failed");
    } finally {
      setSubmitting(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const LANG_OPTIONS: {
    value: Language;
    label: string;
    icon: typeof Globe2;
  }[] = [
    { value: "auto", label: "Auto", icon: Globe2 },
    { value: "ar", label: "العربية", icon: Languages },
    { value: "en", label: "English", icon: Languages },
  ];

  return (
    <div className="flex flex-col h-full" dir="ltr">
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-4 text-center py-20">
            <div className="w-16 h-16 rounded-2xl bg-ash flex items-center justify-center">
              <Mic2 className="w-8 h-8 text-muted" />
            </div>

            <div>
              <p className="text-lg font-medium text-ink">
                Ask about your documents
              </p>
              <p className="text-sm text-muted mt-1">
                Upload files first, then ask anything
              </p>
            </div>
          </div>
        )}

        {messages.map((msg) => {
          const isUser = msg.role === "user";
          const isAr = isArabicText(msg.text);

          return (
            <div
              key={msg.id}
              className={`flex w-full ${
                isUser ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`
                  max-w-[75%] flex flex-col gap-1.5
                  ${isUser ? "items-end" : "items-start"}
                `}
              >
                {isUser ? (
                  <div
                    className={`
                      px-4 py-3 rounded-2xl rounded-br-sm
                      bg-ink text-paper shadow-sm
                      text-[15px] leading-relaxed whitespace-pre-wrap
                      ${isAr ? "text-right font-arabic" : "text-left"}
                    `}
                    dir={isAr ? "rtl" : "ltr"}
                  >
                    {msg.text}
                  </div>
                ) : (
                  <>
                    <AnswerBox text={msg.text} isLoading={!!msg.loading} />
                    {msg.sources && <SourceBox sources={msg.sources} />}
                  </>
                )}
              </div>
            </div>
          );
        })}

        <div ref={bottomRef} />
      </div>

      <div className="border-t border-ash bg-paper/95 backdrop-blur-sm px-4 py-4">
        <div className="flex gap-1 mb-3 justify-start">
          {LANG_OPTIONS.map((opt) => {
            const Icon = opt.icon;

            return (
              <button
                key={opt.value}
                type="button"
                onClick={() => setLanguage(opt.value)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all
                  ${
                    language === opt.value
                      ? "bg-ink text-paper shadow-sm"
                      : "bg-ash text-muted hover:text-ink hover:bg-ash/80"
                  }`}
              >
                <Icon className="w-3.5 h-3.5" />
                {opt.label}
              </button>
            );
          })}
        </div>

        <form onSubmit={handleSubmit} className="flex gap-2 items-end">
          <VoiceRecorder
            language={language}
            disabled={submitting}
            onResult={async (blob) => {
              await handleVoice(blob);
            }}
          />

          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={submitting}
              placeholder="Ask a question about your documents... (Enter to send)"
              rows={1}
              className={`
                w-full resize-none rounded-xl border border-ash bg-white
                px-4 py-3 text-[15px] leading-relaxed text-ink
                placeholder-muted outline-none
                focus:border-teal focus:ring-2 focus:ring-teal/20
                disabled:opacity-50
                max-h-40 overflow-y-auto
                ${isArabicText(query) ? "text-right font-arabic" : "text-left"}
              `}
              dir={isArabicText(query) ? "rtl" : "ltr"}
            />
          </div>

          <button
            type="submit"
            disabled={!query.trim() || submitting}
            className="w-10 h-10 rounded-xl bg-ink text-paper flex items-center justify-center
              hover:bg-ink/80 disabled:opacity-40 disabled:cursor-not-allowed
              transition-all flex-shrink-0"
          >
            {submitting ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <SendHorizontal className="w-4 h-4" />
            )}
          </button>
        </form>
      </div>
    </div>
  );
}