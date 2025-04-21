// frontend/src/components/Footer.tsx
import React from 'react';
import { motion } from 'framer-motion';
import { FiExternalLink } from 'react-icons/fi'; // Add icon for external link

const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-background border-t border-border-color/50 py-10 md:py-12 mt-16"> {/* Softened border color */}
      <div className="container mx-auto px-4 text-center">
        <p className="text-text-muted text-sm mb-3">
          © {currentYear} OULAD Course Recommendation System - Mohit Bhimrajka
        </p>
         <motion.div
             className="mt-4 space-x-6 flex flex-wrap justify-center items-center gap-y-2" // Increased spacing, allow wrapping
             initial={{ opacity: 0 }}
             animate={{ opacity: 1 }}
             transition={{ delay: 0.2 }}
         >
             <a
                 href="https://github.com/mohitbhimrajka/recsys_final"
                 target="_blank"
                 rel="noopener noreferrer"
                 className="inline-flex items-center gap-1 text-text-muted hover:text-primary text-xs transition-colors duration-200 hover:underline outline-none focus-visible:text-primary focus-visible:underline focus-visible:ring-1 focus-visible:ring-primary rounded" // Applied focus-visible
             >
                 View Project on GitHub <FiExternalLink className="flex-shrink-0"/>
             </a>
             <a
                 href="https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad" // Added link to OULAD source
                 target="_blank"
                 rel="noopener noreferrer"
                 className="inline-flex items-center gap-1 text-text-muted hover:text-primary text-xs transition-colors duration-200 hover:underline outline-none focus-visible:text-primary focus-visible:underline focus-visible:ring-1 focus-visible:ring-primary rounded" // Applied focus-visible
             >
                 OULAD Dataset Source <FiExternalLink className="flex-shrink-0"/>
             </a>
         </motion.div>
      </div>
    </footer>
  );
};

export default Footer;