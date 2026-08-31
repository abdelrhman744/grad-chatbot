"use client";

interface Props {
  text: string;
  isLoading: boolean;
  isAr?: boolean;
  statusText?: string;
  streaming?: boolean;
}

const isArabicText = (text: string) => /[؀-ۿ]/.test(text);

export default function AnswerBox({ text, isLoading, isAr, statusText, streaming }: Props) {
  if (isLoading) {
    const rtl = isAr || (statusText ? isArabicText(statusText) : false);
    return (
      <div
        className="flex items-center gap-2 px-4 py-3 rounded-2xl rounded-bl-md bg-surface border border-border shadow-card w-fit"
        dir={rtl ? "rtl" : "ltr"}
      >
        <span className="flex items-center gap-1.5">
          <span className="typing-dot" />
          <span className="typing-dot" />
          <span className="typing-dot" />
        </span>
        {statusText && (
          <span className={`text-xs text-muted ${rtl ? "font-arabic" : ""}`}>{statusText}</span>
        )}
      </div>
    );
  }

  if (!text) return null;

  const rtl = isAr || isArabicText(text);

  return (
    <div
      className={`
        w-fit max-w-full
        px-5 py-4 rounded-2xl rounded-bl-md
        bg-surface border border-border shadow-card
        text-[15px] leading-relaxed whitespace-pre-wrap animate-fade-up
        ${rtl ? "text-right font-arabic" : "text-left"}
      `}
      dir={rtl ? "rtl" : "ltr"}
    >
      {text}
      {streaming && <span className="stream-cursor" />}
    </div>
  );
}
