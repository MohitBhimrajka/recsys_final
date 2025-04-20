// frontend/tailwind.config.js
const defaultTheme = require('tailwindcss/defaultTheme');

/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', ...defaultTheme.fontFamily.sans],
      },
      colors: {
        // True Black Theme Palette
        'background': '#000000', // Pure Black
        'surface': '#111111',   // Very Dark Gray (subtle contrast)
        'primary': {          // Accent Color (Keep Cyan)
          light: '#67e8f9',   // cyan-300
          DEFAULT: '#06b6d4', // cyan-500
          dark: '#0e7490',   // cyan-700
        },
        'secondary': {        // Secondary Accent (Optional - e.g., a subtle purple/violet)
          light: '#a78bfa',  // violet-400
          DEFAULT: '#8b5cf6', // violet-500
          dark: '#7c3aed',   // violet-600
        },
        'text-primary': '#ffffff', // Pure White (max contrast on black)
        'text-secondary': '#e5e7eb', // Off-white (Tailwind gray-200)
        'text-muted': '#9ca3af',     // Muted gray (Tailwind gray-400)
        'border-color': '#2d2d2d', // Dark Gray Border (slightly lighter than surface)
      },
      // Add/keep animation keyframes
      keyframes: {
          fadeInUp: {
              '0%': { opacity: '0', transform: 'translateY(20px)' },
              '100%': { opacity: '1', transform: 'translateY(0)' },
          },
          fadeIn: {
             '0%': { opacity: '0' },
              '100%': { opacity: '1' },
          },
          // Example: Subtle background gradient animation for Hero?
          gradientShift: {
            '0%, 100%': { backgroundPosition: '0% 50%' },
            '50%': { backgroundPosition: '100% 50%' },
          }
      },
      animation: {
          fadeInUp: 'fadeInUp 0.6s ease-out forwards',
          fadeIn: 'fadeIn 0.5s ease-out forwards',
          gradientShift: 'gradientShift 15s ease infinite', // Example usage
      }
    },
  },
  plugins: [],
}