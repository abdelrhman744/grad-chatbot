"use client";

interface Props {
  text: string;
  isLoading: boolean;
  isAr?: boolean;
}

const isArabicText = (text: string) => /[\u0600-\u06FF]/.test(text);

export default function AnswerBox({ text, isLoading, isAr }: Props) {
  if (isLoading) {
    return (
      <div className="flex items-center gap-1.5 px-4 py-3 rounded-2xl rounded-bl-sm bg-white border border-ash shadow-sm w-fit">
        <span className="typing-dot" />
        <span className="typing-dot" />
        <span className="typing-dot" />
      </div>
    );
  }

  if (!text) return null;

  const rtl = isAr || isArabicText(text);

  return (
    <div
      className={`
        w-fit max-w-full
        px-5 py-4 rounded-2xl rounded-bl-sm
        bg-white border border-ash shadow-sm
        text-[15px] leading-relaxed whitespace-pre-wrap animate-fade-up
        ${rtl ? "text-right font-arabic" : "text-left"}
      `}
      dir={rtl ? "rtl" : "ltr"}
    >
      {text}
    </div>
  );
}