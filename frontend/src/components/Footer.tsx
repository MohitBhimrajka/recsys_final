// frontend/src/components/Footer.tsx
import React from 'react';

const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();

  return (
    // Use darker surface or background, adjust border
    <footer className="bg-surface border-t border-border-color py-8 mt-16">
      <div className="container mx-auto px-4 text-center">
        <p className="text-text-muted text-sm">
          © {currentYear} OULAD Course Recommendation System - Mohit Bhimrajka
        </p>
        {/* Optional: Add links */}
         <div className="mt-3 space-x-4">
             {/* Replace '#' with actual links if you have them */}
             <a href="https://github.com/mohitbhimrajka/recsys_final" target="_blank" rel="noopener noreferrer" className="text-text-muted hover:text-primary text-xs">GitHub Repo</a>
             {/* <span className="text-text-muted">|</span>
             <a href="#" className="text-text-muted hover:text-primary text-xs">LinkedIn</a> */}
         </div>
      </div>
    </footer>
  );
};

export default Footer;