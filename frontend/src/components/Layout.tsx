// frontend/src/components/Layout.tsx
import React from 'react';
import { Outlet } from 'react-router-dom';
import Navbar from './Navbar';

const Layout: React.FC = () => {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      {/* Increase max-width here */}
      <main className="flex-grow container mx-auto p-4 pt-8 max-w-7xl"> {/* Changed max-w-4xl to max-w-7xl */}
        <Outlet />
      </main>
       <footer className="bg-gray-800 text-center py-4 mt-auto">
        <p className="text-gray-400 text-sm">
            OULAD Course Recommendation System - Mohit Bhimrajka
        </p>
      </footer>
    </div>
  );
};

export default Layout;