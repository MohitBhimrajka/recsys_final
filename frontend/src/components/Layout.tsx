// frontend/src/components/Layout.tsx
import React from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import Navbar from './Navbar';
import Footer from './Footer'; // Import the new Footer

const Layout: React.FC = () => {
  const location = useLocation();
  // Determine if the current page is the homepage to allow full-width sections if needed
  const isHomePage = location.pathname === '/';

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Navbar />
      {/*
        Main content area.
        If it's the homepage, we don't enforce max-width here,
        allowing sections within HomePage to control their width.
        Otherwise, apply container and padding.
      */}
      <main className={`flex-grow ${isHomePage ? '' : 'container mx-auto px-4 pt-8 max-w-7xl'}`}>
        <Outlet /> {/* Renders the matched child route component */}
      </main>
      <Footer />
    </div>
  );
};

export default Layout;