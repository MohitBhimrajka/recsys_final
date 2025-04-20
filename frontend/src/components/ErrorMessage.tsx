// frontend/src/components/ErrorMessage.tsx
import React from 'react';
import { motion } from 'framer-motion'; // Add animation

interface ErrorMessageProps {
  message: string;
}

const ErrorMessage: React.FC<ErrorMessageProps> = ({ message }) => {
  // Use red shades that work okay on dark background
  const errorBgColor = "bg-red-900"; // Darker red background
  const errorBorderColor = "border-red-600";
  const errorTextColor = "text-red-100"; // Lighter red text
  const errorAccentColor = "text-red-400"; // For icon/bold text

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
      className={`${errorBgColor} border-l-4 ${errorBorderColor} ${errorTextColor} p-4 rounded-md shadow-md my-6 max-w-xl mx-auto`}
      role="alert"
    >
      <div className="flex">
        <div className="py-1">
          {/* SVG Icon - adjusted color */}
          <svg className={`fill-current h-6 w-6 ${errorAccentColor} mr-4`} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
            <path d="M2.93 17.07A10 10 0 1 1 17.07 2.93 10 10 0 0 1 2.93 17.07zM11.4 5.4a1.4 1.4 0 1 0-2.8 0 1.4 1.4 0 0 0 2.8 0zM10 15a1 1 0 0 0 1-1v-4a1 1 0 0 0-2 0v4a1 1 0 0 0 1 1z"/>
          </svg>
        </div>
        <div>
          <p className={`font-bold ${errorAccentColor}`}>Error</p>
          <p className="text-sm">{message}</p>
        </div>
      </div>
    </motion.div>
  );
};

export default ErrorMessage;