import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["'IBM Plex Sans'", "sans-serif"],
        mono: ["'IBM Plex Mono'", "monospace"],
        arabic: ["'Cairo'", "sans-serif"],
      },
      colors: {
        ink:    "#0D0D0D",
        paper:  "#F7F5F0",
        ash:    "#E8E4DC",
        muted:  "#9B9590",
        accent: "#C9472F",
        teal:   "#1D7A72",
        gold:   "#C8922A",
      },
      animation: {
        "fade-up":    "fadeUp 0.4s ease forwards",
        "pulse-dot":  "pulseDot 1.4s infinite",
        "spin-slow":  "spin 2s linear infinite",
      },
      keyframes: {
        fadeUp: {
          "0%":   { opacity: "0", transform: "translateY(12px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        pulseDot: {
          "0%, 80%, 100%": { opacity: "0.2", transform: "scale(0.8)" },
          "40%":           { opacity: "1",   transform: "scale(1.2)" },
        },
      },
    },
  },
  plugins: [],
};
export default config;
