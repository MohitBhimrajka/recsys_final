// frontend/src/components/ErrorMessage.tsx
// No significant changes needed for Phase 2, just ensuring styling is consistent with theme.
import React from 'react';
import { motion } from 'framer-motion';
import { FiAlertTriangle } from 'react-icons/fi'; // Using a different icon

interface ErrorMessageProps {
  message: string;
  title?: string; // Optional title
}

const ErrorMessage: React.FC<ErrorMessageProps> = ({ message, title = "Request Error" }) => {
  // Using Tailwind theme colors directly
  const errorBgColor = "bg-red-900/80"; // Slightly transparent bg
  const errorBorderColor = "border-red-600";
  const errorTextColor = "text-red-100";
  const errorAccentColor = "text-red-400";

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
      className={`${errorBgColor} border ${errorBorderColor} ${errorTextColor} p-4 rounded-lg shadow-md my-6 max-w-xl mx-auto backdrop-blur-sm`} // Use regular border, backdrop blur
      role="alert"
    >
      <div className="flex">
        <div className="py-1">
          <FiAlertTriangle className={`h-6 w-6 ${errorAccentColor} mr-3 flex-shrink-0`} /> {/* Icon */}
        </div>
        <div>
          <p className={`font-semibold ${errorAccentColor} mb-1`}>{title}</p> {/* Use dynamic Title */}
          <p className="text-sm">{message}</p>
        </div>
      </div>
    </motion.div>
  );
};

export default ErrorMessage;