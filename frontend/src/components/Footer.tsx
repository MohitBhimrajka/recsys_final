// frontend/src/components/Footer.tsx
import React from 'react';
import { motion } from 'framer-motion';

const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-background border-t border-border-color py-10 md:py-12 mt-16"> {/* Increased padding */}
      <div className="container mx-auto px-4 text-center">
        <p className="text-text-muted text-sm mb-3">
          © {currentYear} OULAD Course Recommendation System - Mohit Bhimrajka
        </p>
         <motion.div
             className="mt-3 space-x-4"
             initial={{ opacity: 0 }}
             animate={{ opacity: 1 }}
             transition={{ delay: 0.2 }}
         >
             <a
                 href="https://github.com/mohitbhimrajka/recsys_final"
                 target="_blank"
                 rel="noopener noreferrer"
                 className="text-text-muted hover:text-primary text-xs transition-colors duration-200 hover:underline focus:outline-none focus-visible:text-primary focus-visible:underline" // Enhanced hover/focus
             >
                 View Project on GitHub
             </a>
         </motion.div>
      </div>
    </footer>
  );
};

export default Footer;