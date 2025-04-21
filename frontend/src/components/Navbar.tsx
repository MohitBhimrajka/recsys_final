// frontend/src/components/Navbar.tsx
import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { FiCode, FiGithub, FiMenu, FiX } from 'react-icons/fi';
import { motion, AnimatePresence, useScroll, useMotionValueEvent } from 'framer-motion';

const Navbar: React.FC = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);

  const { scrollY } = useScroll();
  useMotionValueEvent(scrollY, "change", (latest) => { setIsScrolled(latest > 10); });

  const linkBaseClasses = "relative px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200 flex items-center gap-1.5 outline-none";
  const linkFocusVisibleClasses = "focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-background";
  const activeLinkClasses = "bg-primary/15 text-primary font-semibold";
  const inactiveLinkClasses = "text-text-secondary hover:bg-surface hover:text-text-primary";

  const getNavLinkClass = ({ isActive }: { isActive: boolean }) => {
    const activeStateClass = isActive ? activeLinkClasses : inactiveLinkClasses;
    return `${linkBaseClasses} ${activeStateClass} ${linkFocusVisibleClasses} group`;
  };

  const Underline = () => (
      <motion.div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" layoutId="underline" initial={false} transition={{ type: 'spring', stiffness: 350, damping: 30 }} />
  );

  const mobileLinkBaseClasses = "block px-3 py-3 rounded-md text-base font-medium outline-none"; // Slightly increased py
  const mobileLinkFocusVisibleClasses = "focus-visible:bg-surface focus-visible:text-primary focus-visible:ring-1 focus-visible:ring-primary";
  const mobileInactiveLinkClass = `text-text-secondary hover:bg-surface hover:text-text-primary ${mobileLinkFocusVisibleClasses}`;
  const mobileActiveLinkClass = `bg-primary/15 text-primary font-semibold ${mobileLinkFocusVisibleClasses}`;

   const getMobileNavLinkClass = ({ isActive }: { isActive: boolean }) =>
    `${mobileLinkBaseClasses} ${isActive ? mobileActiveLinkClass : mobileInactiveLinkClass}`;

  const toggleMobileMenu = () => setIsMobileMenuOpen(!isMobileMenuOpen);

  const mobileMenuVariants = {
    hidden: { opacity: 0, y: -10 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.3, ease: 'easeOut'} },
    exit: { opacity: 0, y: -5, transition: { duration: 0.2, ease: 'easeIn'} }
  };

  return (
    <motion.nav className={`sticky top-0 z-50 transition-all duration-300 ${isScrolled ? 'bg-surface/90 backdrop-blur-lg shadow-lg border-b border-border-color/50' : 'bg-transparent border-b border-transparent'}`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Left Side: Brand */}
          <div className="flex-shrink-0">
            <NavLink to="/" className="text-white text-lg hover:text-primary transition-colors duration-150 flex items-center font-semibold outline-none focus-visible:ring-2 focus-visible:ring-primary rounded" title="Homepage">
               OULAD Recommendation System
            </NavLink>
          </div>

          {/* Right Side: Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-1">
             <motion.div layout className="relative">
                 <NavLink to="/" end className={getNavLinkClass}>
                     {({ isActive }) => ( <> Home {isActive && <Underline />} </> )}
                 </NavLink>
             </motion.div>
             <motion.div layout className="relative">
                 <NavLink to="/demo" className={getNavLinkClass}>
                     {({ isActive }) => ( <> Demo {isActive && <Underline />} </> )}
                 </NavLink>
              </motion.div>
             <motion.div layout className="relative">
                 <NavLink to="/about" className={getNavLinkClass}>
                     {({ isActive }) => ( <> How it Works {isActive && <Underline />} </> )}
                 </NavLink>
              </motion.div>
             <motion.div layout className="relative">
                 <NavLink to="/code-explorer" className={getNavLinkClass}>
                     {({ isActive }) => ( <> <FiCode size={16} /> Code Explorer {isActive && <Underline />} </> )}
                 </NavLink>
              </motion.div>
             <span className={`h-6 w-px mx-3 transition-colors duration-300 ${isScrolled ? 'bg-border-color' : 'bg-transparent'}`} aria-hidden="true"></span>
             <motion.a href="https://github.com/mohitbhimrajka/recsys_final" target="_blank" rel="noopener noreferrer" className="text-text-muted hover:text-primary transition-colors duration-200 p-2 rounded-full hover:bg-surface outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-1 focus-visible:ring-offset-background" title="View source on GitHub" whileHover={{ scale: 1.15, rotate: 5 }} whileTap={{ scale: 0.95 }}>
               <FiGithub size={20} /> <span className="sr-only">GitHub</span>
             </motion.a>
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden flex items-center">
             <button onClick={toggleMobileMenu} className="text-text-muted hover:text-white p-2 rounded-md outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-primary" aria-controls="mobile-menu" aria-expanded={isMobileMenuOpen}>
                 <span className="sr-only">Open main menu</span>
                 {isMobileMenuOpen ? <FiX size={24} /> : <FiMenu size={24} />}
             </button>
          </div>
        </div>
      </div>

       {/* Mobile Menu Panel */}
       <AnimatePresence>
           {isMobileMenuOpen && (
                <motion.div id="mobile-menu" className={`md:hidden border-t bg-surface shadow-lg ${isScrolled ? 'border-border-color/50' : 'border-transparent'}`} variants={mobileMenuVariants} initial="hidden" animate="visible" exit="exit" style={{ overflow: 'hidden' }}>
                    <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
                        <NavLink to="/" end className={getMobileNavLinkClass} onClick={toggleMobileMenu}>Home</NavLink>
                        <NavLink to="/demo" className={getMobileNavLinkClass} onClick={toggleMobileMenu}>Demo</NavLink>
                        <NavLink to="/about" className={getMobileNavLinkClass} onClick={toggleMobileMenu}>How it Works</NavLink>
                        <NavLink to="/code-explorer" className={getMobileNavLinkClass} onClick={toggleMobileMenu}>Code Explorer</NavLink>
                         <a href="https://github.com/mohitbhimrajka/recsys_final" target="_blank" rel="noopener noreferrer" className={`${mobileLinkBaseClasses} ${mobileInactiveLinkClass} flex items-center gap-2`} onClick={toggleMobileMenu}>
                            <FiGithub size={18} /> GitHub
                         </a>
                    </div>
                </motion.div>
           )}
        </AnimatePresence>
    </motion.nav>
  );
};

export default Navbar;