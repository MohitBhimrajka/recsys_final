// frontend/src/components/Layout.tsx
import React from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import Navbar from './Navbar';
import Footer from './Footer';
import { motion, AnimatePresence } from 'framer-motion';

const Layout: React.FC = () => {
  const location = useLocation();
  const isHomePage = location.pathname === '/';

  // Simple page transition variant
  const pageVariants = {
    initial: { opacity: 0, y: 10 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.4, ease: 'easeInOut' } },
    exit: { opacity: 0, y: -5, transition: { duration: 0.2, ease: 'easeIn' } }
  };

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Navbar />
      {/* AnimatePresence for page transitions */}
      <AnimatePresence mode="wait">
        {/* Use location.pathname as key to trigger transition on route change */}
        <motion.main
          key={location.pathname}
          className={`flex-grow ${isHomePage ? '' : 'container mx-auto px-4 pt-8 max-w-7xl'}`}
          variants={pageVariants}
          initial="initial"
          animate="animate"
          exit="exit"
        >
          <Outlet />
        </motion.main>
      </AnimatePresence>
      <Footer />
    </div>
  );
};

export default Layout;