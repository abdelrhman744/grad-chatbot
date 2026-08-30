"use client";

import { LucideIcon } from "lucide-react";

interface Props {
  icon: LucideIcon;
  title: string;
  subtitle?: string;
  className?: string;
}

export default function EmptyState({ icon: Icon, title, subtitle, className = "" }: Props) {
  return (
    <div className={`flex flex-col items-center justify-center gap-3 text-center py-10 ${className}`}>
      <div className="w-14 h-14 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center">
        <Icon className="w-7 h-7 text-primary-light" strokeWidth={1.75} />
      </div>
      <div>
        <p className="text-sm font-medium text-ink">{title}</p>
        {subtitle && <p className="text-xs text-muted mt-1 max-w-[240px] leading-relaxed">{subtitle}</p>}
      </div>
    </div>
  );
}
