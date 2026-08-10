/** @type {import('tailwindcss').Config} */

export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    container: {
      center: true,
    },
    extend: {
      colors: {
        amber: {
          accent: "#D4A853",
          light: "#E8C878",
          dark: "#B8922F",
          glow: "rgba(212, 168, 83, 0.15)",
        },
        dark: {
          950: "#0A0A0B",
          900: "#111113",
          850: "#161618",
          800: "#1C1C1F",
          700: "#2A2A2E",
          600: "#3A3A40",
          500: "#52525A",
        },
        ivory: {
          DEFAULT: "#F5F0E8",
          muted: "#A8A29E",
          dim: "#78716C",
        },
      },
      fontFamily: {
        display: ["Playfair Display", "Georgia", "serif"],
        body: ["DM Sans", "system-ui", "sans-serif"],
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "fade-in": "fadeIn 0.5s ease-out",
        "slide-up": "slideUp 0.4s ease-out",
        "glow": "glow 2s ease-in-out infinite alternate",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(16px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        glow: {
          "0%": { boxShadow: "0 0 8px rgba(212, 168, 83, 0.2)" },
          "100%": { boxShadow: "0 0 20px rgba(212, 168, 83, 0.4)" },
        },
      },
    },
  },
  plugins: [],
};
