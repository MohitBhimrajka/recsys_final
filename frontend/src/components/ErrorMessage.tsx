// frontend/src/components/ErrorMessage.tsx
import React from 'react';

interface ErrorMessageProps {
  message: string;
}

const ErrorMessage: React.FC<ErrorMessageProps> = ({ message }) => {
  return (
    <div
      className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded-md shadow-md my-6 max-w-xl mx-auto" // Centered, max-width
      role="alert"
    >
      <div className="flex items-center">
        <div className="py-1">
          {/* Optional: Add an SVG icon here */}
          <svg className="fill-current h-6 w-6 text-red-500 mr-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M2.93 17.07A10 10 0 1 1 17.07 2.93 10 10 0 0 1 2.93 17.07zM11.4 5.4a1.4 1.4 0 1 0-2.8 0 1.4 1.4 0 0 0 2.8 0zM10 15a1 1 0 0 0 1-1v-4a1 1 0 0 0-2 0v4a1 1 0 0 0 1 1z"/></svg>
        </div>
        <div>
          <p className="font-bold">Error</p>
          <p className="text-sm">{message}</p>
        </div>
      </div>
    </div>
  );
};

export default ErrorMessage;