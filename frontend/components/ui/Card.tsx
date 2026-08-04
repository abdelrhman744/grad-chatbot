"use client";

import { HTMLAttributes, forwardRef } from "react";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  hoverable?: boolean;
  glass?: boolean;
  padding?: "none" | "sm" | "md" | "lg";
}

const PADDING: Record<NonNullable<CardProps["padding"]>, string> = {
  none: "",
  sm: "p-3",
  md: "p-4",
  lg: "p-5",
};

const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ className = "", hoverable = false, glass = false, padding = "md", children, ...rest }, ref) => (
    <div
      ref={ref}
      className={`
        rounded-2xl border border-border shadow-card
        transition-all duration-200
        ${glass ? "glass" : "bg-surface"}
        ${hoverable ? "hover:border-primary/40 hover:shadow-glow hover:-translate-y-0.5 cursor-pointer" : ""}
        ${PADDING[padding]}
        ${className}
      `}
      {...rest}
    >
      {children}
    </div>
  )
);
Card.displayName = "Card";

export default Card;
