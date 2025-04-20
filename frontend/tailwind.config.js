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
        'background': '#000000',
        'surface': '#111111',
        'primary': {
          light: '#67e8f9',
          DEFAULT: '#06b6d4',
          dark: '#0e7490',
        },
        'secondary': {
          light: '#a78bfa',
          DEFAULT: '#8b5cf6',
          dark: '#7c3aed',
        },
        'text-primary': '#ffffff',
        'text-secondary': '#e5e7eb',
        'text-muted': '#9ca3af',
        'border-color': '#2d2d2d',
      },
      keyframes: {
          fadeInUp: { '0%': { opacity: '0', transform: 'translateY(20px)' }, '100%': { opacity: '1', transform: 'translateY(0)' }, },
          fadeIn: { '0%': { opacity: '0' }, '100%': { opacity: '1' }, },
          gradientShift: { '0%, 100%': { backgroundPosition: '0% 50%' }, '50%': { backgroundPosition: '100% 50%' }, }
      },
      animation: {
          fadeInUp: 'fadeInUp 0.6s ease-out forwards',
          fadeIn: 'fadeIn 0.5s ease-out forwards',
          gradientShift: 'gradientShift 15s ease infinite',
      }
    },
  },
  plugins: [],
}