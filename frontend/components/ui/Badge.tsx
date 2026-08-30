"use client";

import { LucideIcon } from "lucide-react";

export type BadgeColor = "primary" | "accent" | "success" | "warning" | "danger" | "muted";

interface Props {
  children: React.ReactNode;
  color?: BadgeColor;
  icon?: LucideIcon;
  className?: string;
}

const COLOR_CLASSES: Record<BadgeColor, string> = {
  primary: "bg-primary/15 text-primary-light",
  accent:  "bg-accent/15 text-accent",
  success: "bg-success/15 text-success",
  warning: "bg-warning/15 text-warning",
  danger:  "bg-danger/15 text-danger",
  muted:   "bg-surface-2 text-muted",
};

export default function Badge({ children, color = "muted", icon: Icon, className = "" }: Props) {
  return (
    <span
      className={`
        inline-flex items-center gap-1 rounded-full px-2.5 py-1
        text-xs font-medium whitespace-nowrap
        ${COLOR_CLASSES[color]} ${className}
      `}
    >
      {Icon && <Icon className="w-3 h-3 flex-shrink-0" />}
      {children}
    </span>
  );
}
