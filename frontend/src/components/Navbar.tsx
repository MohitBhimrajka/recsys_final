// frontend/src/components/Navbar.tsx
import React from 'react';
import { NavLink } from 'react-router-dom';
import { FiCode, FiGithub, FiMenu } from 'react-icons/fi';
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const Navbar: React.FC = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  // Shared classes for NavLink
  const linkBaseClasses = "px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200 flex items-center gap-1.5";
  const activeLinkClasses = "bg-primary/10 text-primary";
  const inactiveLinkClasses = "text-text-secondary hover:bg-surface hover:text-text-primary";

  const getNavLinkClass = ({ isActive }: { isActive: boolean }) =>
    `${linkBaseClasses} ${isActive ? activeLinkClasses : inactiveLinkClasses}`;

  const mobileLinkClass = "block px-3 py-2 rounded-md text-base font-medium text-text-secondary hover:bg-surface hover:text-text-primary";
  const mobileActiveLinkClass = "block px-3 py-2 rounded-md text-base font-medium bg-primary/10 text-primary";

   const getMobileNavLinkClass = ({ isActive }: { isActive: boolean }) =>
    `${isActive ? mobileActiveLinkClass : mobileLinkClass}`;


  const toggleMobileMenu = () => setIsMobileMenuOpen(!isMobileMenuOpen);

  // Animation variants for mobile menu
  const mobileMenuVariants = {
    hidden: { opacity: 0, y: -20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.2, ease: 'easeOut'} },
    exit: { opacity: 0, y: -10, transition: { duration: 0.15, ease: 'easeIn'} }
  };

  return (
    <nav className="bg-surface/95 backdrop-blur-sm shadow-md sticky top-0 z-50 border-b border-border-color">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">

          {/* Left Side: Brand/Logo */}
          <div className="flex-shrink-0">
            <NavLink
                to="/"
                className="text-white font-semibold text-lg hover:text-gray-200 transition-colors duration-150 flex items-center"
                title="Homepage"
            >
               OULAD Recommendation System
            </NavLink>
          </div>

          {/* Right Side: Navigation Links & Actions (Desktop) */}
          <div className="hidden md:flex items-center space-x-1">
             <NavLink to="/" end className={getNavLinkClass}>Home</NavLink>
             <NavLink to="/demo" className={getNavLinkClass}>Demo</NavLink>
             <NavLink to="/about" className={getNavLinkClass}>How it Works</NavLink>
             <NavLink to="/code-explorer" className={getNavLinkClass}>
               <FiCode size={16} /> Code Explorer
             </NavLink>
             <span className="h-6 w-px bg-border-color mx-3" aria-hidden="true"></span>
             <a
                href="https://github.com/mohitbhimrajka/recsys_final"
                target="_blank"
                rel="noopener noreferrer"
                className="text-text-muted hover:text-primary transition-colors duration-200 p-2 rounded-full hover:bg-surface"
                title="View source on GitHub"
             >
               <FiGithub size={20} />
                <span className="sr-only">GitHub</span>
             </a>
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden flex items-center">
             <button
                onClick={toggleMobileMenu}
                className="text-text-muted hover:text-white p-2 rounded-md focus:outline-none focus:ring-2 focus:ring-inset focus:ring-primary"
                aria-controls="mobile-menu"
                aria-expanded={isMobileMenuOpen}
             >
                 <span className="sr-only">Open main menu</span>
                 <FiMenu size={24} />
             </button>
          </div>

        </div>
      </div>

       {/* Mobile Menu Panel */}
       <AnimatePresence>
           {isMobileMenuOpen && (
                <motion.div
                    id="mobile-menu"
                    className="md:hidden border-t border-border-color bg-surface shadow-lg" // Added background
                    variants={mobileMenuVariants}
                    initial="hidden"
                    animate="visible"
                    exit="exit"
                 >
                    <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
                        <NavLink to="/" end className={getMobileNavLinkClass} onClick={toggleMobileMenu}>Home</NavLink>
                        <NavLink to="/demo" className={getMobileNavLinkClass} onClick={toggleMobileMenu}>Demo</NavLink>
                        <NavLink to="/about" className={getMobileNavLinkClass} onClick={toggleMobileMenu}>How it Works</NavLink>
                        <NavLink to="/code-explorer" className={getMobileNavLinkClass} onClick={toggleMobileMenu}>Code Explorer</NavLink>
                         <a
                            href="https://github.com/mohitbhimrajka/recsys_final"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="block px-3 py-2 rounded-md text-base font-medium text-text-secondary hover:bg-surface hover:text-text-primary flex items-center gap-2"
                            onClick={toggleMobileMenu}
                         >
                            <FiGithub size={18} /> GitHub
                         </a>
                    </div>
                </motion.div>
           )}
        </AnimatePresence>
    </nav>
  );
};

export default Navbar;