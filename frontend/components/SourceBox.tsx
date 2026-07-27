"use client";

interface Props {
  sources: string;
}

export default function SourceBox({ sources }: Props) {
  if (!sources) return null;

  const rtl = /[\u0600-\u06FF]/.test(sources);

  const parts = sources.includes("|")
    ? sources.replace(/^(Sources:|المصادر:)\s*/i, "").split("|").map((s) => s.trim())
    : [sources];

  const label = rtl ? "المصادر" : "Sources";

  return (
    <div className={`mt-2 animate-fade-up ${rtl ? "text-right" : "text-left"}`} dir={rtl ? "rtl" : "ltr"}>
      <p className="text-xs font-mono text-muted uppercase tracking-wider mb-1.5">{label}</p>
      <div className="flex flex-wrap gap-2">
        {parts.map((src, i) => (
          <span
            key={i}
            className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full
              bg-ash text-ink text-xs font-medium border border-ash/80"
          >
            <svg className="w-3 h-3 text-muted flex-shrink-0" fill="none" viewBox="0 0 24 24"
              stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414A1 1 0 0119 9.414V19a2 2 0 01-2 2z"/>
            </svg>
            {src}
          </span>
        ))}
      </div>
    </div>
  );
}
