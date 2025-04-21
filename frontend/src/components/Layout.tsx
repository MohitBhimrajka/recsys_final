// frontend/src/components/Layout.tsx
import React from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import Navbar from './Navbar';
import Footer from './Footer';
import { motion, AnimatePresence } from 'framer-motion';

const Layout: React.FC = () => {
  const location = useLocation();
  const isHomePage = location.pathname === '/';

  // Page transition variant
  const pageVariants = {
    initial: { opacity: 0, y: 15 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.43, 0.13, 0.23, 0.96] } },
    exit: { opacity: 0, y: -10, transition: { duration: 0.3, ease: 'easeIn' } }
  };

  // Subtle fade-in for the main content itself
  const contentVariants = {
    initial: { opacity: 0 },
    animate: { opacity: 1, transition: { duration: 0.4, delay: 0.1 } }, // Slight delay after page transition starts
  };


  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Navbar />
      {/* AnimatePresence for page transitions */}
      <AnimatePresence mode="wait">
        {/* Use location.pathname as key to trigger transition on route change */}
        <motion.div
          key={location.pathname}
          className="flex-grow" // Apply flex-grow here
          variants={pageVariants}
          initial="initial"
          animate="animate"
          exit="exit"
        >
           {/* Add another motion div for content fade-in */}
           <motion.main
               className={`${isHomePage ? '' : 'container mx-auto px-4 pt-8 pb-16 max-w-7xl'}`}
               variants={contentVariants} // Apply content fade-in here
               initial="initial"
               animate="animate"
           >
                <Outlet /> {/* Page content renders here */}
           </motion.main>
        </motion.div>
      </AnimatePresence>
      <Footer />
    </div>
  );
};

export default Layout;