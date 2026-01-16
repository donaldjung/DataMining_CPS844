/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'dm-blue': '#3b82f6',
        'dm-orange': '#f97316',
        'dm-purple': '#8b5cf6',
        'dm-cyan': '#06b6d4',
        'dm-dark': '#0f172a',
        'dm-darker': '#020617',
      },
    },
  },
  plugins: [],
}
