"use client";

interface Props {
  className?: string;
}

export default function Skeleton({ className = "h-4 w-full" }: Props) {
  return <div className={`skeleton-shimmer animate-shimmer rounded-md ${className}`} />;
}
