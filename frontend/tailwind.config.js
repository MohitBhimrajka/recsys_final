// frontend/tailwind.config.js
const defaultTheme = require('tailwindcss/defaultTheme') // Import default theme

/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
        // Add custom font family (optional, requires font import)
        fontFamily: {
            sans: ['Inter', ...defaultTheme.fontFamily.sans], // Set Inter as default sans-serif
        },
        // Define custom colors (optional, use Tailwind defaults or define your own)
        colors: {
            'primary': {
                light: '#67e8f9', // cyan-300
                DEFAULT: '#06b6d4', // cyan-500
                dark: '#0e7490', // cyan-700
            },
            'secondary': {
                light: '#f0f9ff', // sky-50
                DEFAULT: '#38bdf8', // sky-400
                dark: '#0369a1', // sky-700
            },
            // Add more custom colors if needed
        },
    },
  },
  plugins: [],
}