import type { Config } from "tailwindcss";

const token = (name: string) => `oklch(var(--${name}) / <alpha-value>)`;

const config: Config = {
  darkMode: "media",
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: token("background"),
        foreground: token("foreground"),
        card: token("card"),
        border: token("border"),
        muted: { DEFAULT: token("muted"), foreground: token("muted-foreground") },
        primary: { DEFAULT: token("primary"), foreground: token("primary-foreground") },
        destructive: token("destructive"),
        success: token("success"),
      },
      fontFamily: {
        sans: ["var(--font-sans)", "var(--font-fa)", "system-ui", "sans-serif"],
        fa: ["var(--font-fa)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      keyframes: {
        pulsering: {
          "0%": { boxShadow: "0 0 0 0 oklch(var(--destructive) / 0.45)" },
          "100%": { boxShadow: "0 0 0 14px oklch(var(--destructive) / 0)" },
        },
      },
      animation: { pulsering: "pulsering 1.4s ease-out infinite" },
    },
  },
  plugins: [],
};

export default config;
