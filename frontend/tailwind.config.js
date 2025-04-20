// frontend/tailwind.config.js
const defaultTheme = require('tailwindcss/defaultTheme');

/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}", // Scans these files for Tailwind classes
  ],
  theme: {
    extend: {
      // Add custom fonts or extend existing ones
      fontFamily: {
        sans: ['Inter', ...defaultTheme.fontFamily.sans], // Use Inter as the default sans-serif
      },
      // Define custom colors for the dark theme
      colors: {
        'background': '#000000', // Pure black background
        'surface': '#111111',    // Slightly lighter surface color
        'primary': {
          light: '#67e8f9',    // Lighter cyan (Cyan 300)
          DEFAULT: '#06b6d4',  // Default primary color (Cyan 600)
          dark: '#0e7490',     // Darker cyan (Cyan 800)
        },
        'secondary': {
          light: '#a78bfa',    // Lighter violet (Violet 400)
          DEFAULT: '#8b5cf6',  // Default secondary color (Violet 500)
          dark: '#7c3aed',     // Darker violet (Violet 600)
        },
        'text-primary': '#ffffff',   // Main text color (white)
        'text-secondary': '#e5e7eb', // Slightly muted text (Gray 200)
        'text-muted': '#9ca3af',     // Muted text (Gray 400)
        'border-color': '#2d2d2d',   // Border color for elements
      },
      // Define custom keyframes for animations
      keyframes: {
          fadeInUp: {
             '0%': { opacity: '0', transform: 'translateY(20px)' },
             '100%': { opacity: '1', transform: 'translateY(0)' },
           },
          fadeIn: {
              '0%': { opacity: '0' },
              '100%': { opacity: '1' },
          },
          gradientShift: { // For animated gradients
              '0%, 100%': { backgroundPosition: '0% 50%' },
              '50%': { backgroundPosition: '100% 50%' },
          },
          // Keyframe for border gradient animation (if used with complex setup)
          borderGradientSpin: {
              '100%': { '--border-angle': '360deg' },
          },
          // Subtle pulse animation (can be used for tags etc.)
          pulseSlight: {
              '0%, 100%': { transform: 'scale(1)', opacity: '1' },
              '50%': { transform: 'scale(1.03)', opacity: '0.85' },
          }
      },
      // Define custom animations using the keyframes
      animation: {
          fadeInUp: 'fadeInUp 0.6s ease-out forwards',
          fadeIn: 'fadeIn 0.5s ease-out forwards',
          gradientShift: 'gradientShift 15s ease infinite',
          // Animation using the keyframe (if used)
          borderGradientSpin: 'borderGradientSpin 3s linear infinite',
          pulseSlight: 'pulseSlight 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  // Add plugins
  plugins: [
    require('@tailwindcss/typography'), // Plugin for rich text styling (used in AboutPage)
  ],
}