import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: ["'IBM Plex Sans'", "sans-serif"],
        mono: ["'IBM Plex Mono'", "monospace"],
        arabic: ["'Cairo'", "sans-serif"],
      },
      colors: {
        // Base surfaces — deep navy/slate, layered from darkest (base) to
        // lightest (surface-2) so cards/inputs read as "raised" without
        // needing heavy borders.
        base:       "#0B1220",
        surface:    "#111827",
        "surface-2": "#182235",
        "surface-3": "#1F2A3D",
        border:     "#22304A",
        "border-light": "#2C3B57",

        // Text
        ink:    "#F1F5F9",
        muted:  "#94A3B8",
        subtle: "#64748B",

        // Brand — indigo primary with a cyan accent, matching a modern
        // "AI assistant" product palette (ChatGPT/Claude/Perplexity-adjacent).
        primary: {
          DEFAULT: "#6366F1",
          dark:    "#4F46E5",
          light:   "#818CF8",
        },
        accent:  "#06B6D4",
        success: "#10B981",
        warning: "#F59E0B",
        danger:  "#EF4444",
      },
      backgroundImage: {
        "gradient-primary": "linear-gradient(135deg, #6366F1 0%, #4F46E5 100%)",
        "gradient-accent":  "linear-gradient(135deg, #06B6D4 0%, #6366F1 100%)",
        "gradient-radial-glow":
          "radial-gradient(circle at top, rgba(99,102,241,0.16), transparent 60%)",
        "gradient-surface": "linear-gradient(180deg, #111827 0%, #0B1220 100%)",
      },
      boxShadow: {
        card:  "0 1px 2px rgba(0,0,0,0.24), 0 1px 3px rgba(0,0,0,0.28)",
        hover: "0 12px 28px rgba(0,0,0,0.35)",
        glow:  "0 0 0 1px rgba(99,102,241,0.4), 0 8px 24px rgba(99,102,241,0.25)",
        "glow-accent": "0 0 0 1px rgba(6,182,212,0.4), 0 8px 24px rgba(6,182,212,0.2)",
      },
      animation: {
        "fade-up":     "fadeUp 0.4s ease forwards",
        "fade-in":     "fadeIn 0.2s ease forwards",
        "pulse-dot":   "pulseDot 1.4s infinite",
        "spin-slow":   "spin 2s linear infinite",
        shimmer:       "shimmer 1.8s ease-in-out infinite",
        blink:         "blink 1s step-start infinite",
        "slide-in":    "slideIn 0.25s ease forwards",
        "scale-in":    "scaleIn 0.15s ease forwards",
      },
      keyframes: {
        fadeUp: {
          "0%":   { opacity: "0", transform: "translateY(12px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        fadeIn: {
          "0%":   { opacity: "0" },
          "100%": { opacity: "1" },
        },
        pulseDot: {
          "0%, 80%, 100%": { opacity: "0.2", transform: "scale(0.8)" },
          "40%":           { opacity: "1",   transform: "scale(1.2)" },
        },
        shimmer: {
          "0%":   { backgroundPosition: "-400px 0" },
          "100%": { backgroundPosition: "400px 0" },
        },
        blink: {
          "50%": { opacity: "0" },
        },
        slideIn: {
          "0%":   { opacity: "0", transform: "translateX(-12px)" },
          "100%": { opacity: "1", transform: "translateX(0)" },
        },
        scaleIn: {
          "0%":   { opacity: "0", transform: "scale(0.96)" },
          "100%": { opacity: "1", transform: "scale(1)" },
        },
      },
    },
  },
  plugins: [],
};
export default config;
