// frontend/src/components/ErrorMessage.tsx
import React from 'react';
import { motion } from 'framer-motion';
import { FiAlertTriangle } from 'react-icons/fi'; // Using a different icon

interface ErrorMessageProps {
  message: string;
}

const ErrorMessage: React.FC<ErrorMessageProps> = ({ message }) => {
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
          <FiAlertTriangle className={`h-6 w-6 ${errorAccentColor} mr-3`} /> {/* Icon */}
        </div>
        <div>
          <p className={`font-semibold ${errorAccentColor} mb-1`}>Request Error</p> {/* Updated Title */}
          <p className="text-sm">{message}</p>
        </div>
      </div>
    </motion.div>
  );
};

export default ErrorMessage;