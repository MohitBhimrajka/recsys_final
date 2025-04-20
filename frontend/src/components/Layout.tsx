// frontend/src/components/Layout.tsx
import React from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import Navbar from './Navbar';
import Footer from './Footer';

const Layout: React.FC = () => {
  const location = useLocation();
  const isHomePage = location.pathname === '/';

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Navbar />
      <main className={`flex-grow ${isHomePage ? '' : 'container mx-auto px-4 pt-8 max-w-7xl'}`}>
        <Outlet />
      </main>
      <Footer />
    </div>
  );
};

export default Layout;