// frontend/src/components/Navbar.tsx
import React, { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import { FiCode, FiGithub, FiMenu } from 'react-icons/fi';
import { motion, AnimatePresence, useScroll, useMotionValueEvent } from 'framer-motion';

const Navbar: React.FC = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);

  // Detect scroll position
  const { scrollY } = useScroll();
  useMotionValueEvent(scrollY, "change", (latest) => {
     setIsScrolled(latest > 10); // Set true if scrolled more than 10px
  });

  // Shared classes for NavLink
  const linkBaseClasses = "relative px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200 flex items-center gap-1.5 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-background"; // Added focus style
  // Use slightly more vibrant active state
  const activeLinkClasses = "bg-primary/15 text-primary"; // Changed from /10
  const inactiveLinkClasses = "text-text-secondary hover:bg-surface hover:text-text-primary";

  // Function to generate NavLink classes and add animated underline
  const getNavLinkClass = ({ isActive }: { isActive: boolean }) => {
    const activeStateClass = isActive ? activeLinkClasses : inactiveLinkClasses;
    // Base classes for the link itself
    let classes = `${linkBaseClasses} ${activeStateClass}`;
    // Add classes for the underline pseudo-element (group-hover target)
    classes += " group"; // Add group class for targeting underline on hover
    return classes;
  };
  // Underline element
  const Underline = () => (
      <motion.div
          className="absolute bottom-0 left-0 h-0.5 bg-primary"
          layoutId="underline" // Shared layout ID for smooth transition between links
          initial={false} // Don't animate initially
          transition={{ type: 'spring', stiffness: 350, damping: 30 }} // Spring animation
      />
  );

  const mobileLinkClass = "block px-3 py-2 rounded-md text-base font-medium text-text-secondary hover:bg-surface hover:text-text-primary focus:outline-none focus-visible:bg-surface"; // Added mobile focus
  const mobileActiveLinkClass = "block px-3 py-2 rounded-md text-base font-medium bg-primary/15 text-primary"; // Changed from /10

   const getMobileNavLinkClass = ({ isActive }: { isActive: boolean }) =>
    `${isActive ? mobileActiveLinkClass : mobileLinkClass}`;


  const toggleMobileMenu = () => setIsMobileMenuOpen(!isMobileMenuOpen);

  // Animation variants for mobile menu
  const mobileMenuVariants = {
    hidden: { opacity: 0, height: 0 }, // Animate height for slide effect
    visible: { opacity: 1, height: 'auto', transition: { duration: 0.3, ease: 'easeOut'} },
    exit: { opacity: 0, height: 0, transition: { duration: 0.2, ease: 'easeIn'} }
  };

  return (
    <motion.nav
        className={`sticky top-0 z-50 transition-all duration-300 ${isScrolled ? 'bg-surface/95 backdrop-blur-md shadow-lg border-b border-border-color' : 'bg-transparent border-b border-transparent'}`} // Dynamic background and border based on scroll
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">

          {/* Left Side: Brand/Logo */}
          <div className="flex-shrink-0">
            <NavLink
                to="/"
                className="text-white font-semibold text-lg hover:text-primary transition-colors duration-150 flex items-center focus:outline-none focus-visible:ring-2 focus-visible:ring-primary rounded" // Added focus style
                title="Homepage"
            >
               OULAD RecSys
            </NavLink>
          </div>

          {/* Right Side: Navigation Links & Actions (Desktop) */}
          <div className="hidden md:flex items-center space-x-1">
             {/* Wrap NavLinks in motion.div for layout animation */}
             <motion.div layout className="relative">
                 <NavLink to="/" end className={getNavLinkClass}>
                     {({ isActive }: { isActive: boolean }) => (
                         <>
                             Home
                             <AnimatePresence>
                                 {isActive && <Underline />}
                             </AnimatePresence>
                         </>
                     )}
                 </NavLink>
             </motion.div>
             <motion.div layout className="relative">
                 <NavLink to="/demo" className={getNavLinkClass}>
                     {({ isActive }: { isActive: boolean }) => (
                         <>
                             Demo
                             <AnimatePresence>
                                 {isActive && <Underline />}
                             </AnimatePresence>
                         </>
                     )}
                 </NavLink>
              </motion.div>
             <motion.div layout className="relative">
                 <NavLink to="/about" className={getNavLinkClass}>
                     {({ isActive }: { isActive: boolean }) => (
                         <>
                             How it Works
                             <AnimatePresence>
                                 {isActive && <Underline />}
                             </AnimatePresence>
                         </>
                     )}
                 </NavLink>
              </motion.div>
             <motion.div layout className="relative">
                 <NavLink to="/code-explorer" className={getNavLinkClass}>
                     {({ isActive }: { isActive: boolean }) => (
                         <>
                             <FiCode size={16} /> Code Explorer
                             <AnimatePresence>
                                 {isActive && <Underline />}
                             </AnimatePresence>
                         </>
                     )}
                 </NavLink>
              </motion.div>
             <span className={`h-6 w-px mx-3 transition-colors duration-300 ${isScrolled ? 'bg-border-color' : 'bg-transparent'}`} aria-hidden="true"></span>
             {/* GitHub Icon with Motion */}
             <motion.a
                href="https://github.com/mohitbhimrajka/recsys_final"
                target="_blank"
                rel="noopener noreferrer"
                className="text-text-muted hover:text-primary transition-colors duration-200 p-2 rounded-full hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-1 focus-visible:ring-offset-background" // Added focus style
                title="View source on GitHub"
                whileHover={{ scale: 1.15, rotate: 5 }} // Added rotate
                whileTap={{ scale: 0.95 }}
             >
               <FiGithub size={20} />
                <span className="sr-only">GitHub</span>
             </motion.a>
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden flex items-center">
             <button
                onClick={toggleMobileMenu}
                className="text-text-muted hover:text-white p-2 rounded-md focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-primary" // Use visible focus ring
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
                    className={`md:hidden border-t bg-surface shadow-lg ${isScrolled ? 'border-border-color' : 'border-transparent'}`} // Dynamic border
                    variants={mobileMenuVariants}
                    initial="hidden"
                    animate="visible"
                    exit="exit"
                    // Ensure content doesn't overflow during animation
                    style={{ overflow: 'hidden' }}
                 >
                    {/* Padding applied here */}
                    <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
                        <NavLink to="/" end className={getMobileNavLinkClass} onClick={toggleMobileMenu}>Home</NavLink>
                        <NavLink to="/demo" className={getMobileNavLinkClass} onClick={toggleMobileMenu}>Demo</NavLink>
                        <NavLink to="/about" className={getMobileNavLinkClass} onClick={toggleMobileMenu}>How it Works</NavLink>
                        <NavLink to="/code-explorer" className={getMobileNavLinkClass} onClick={toggleMobileMenu}>Code Explorer</NavLink>
                         <a
                            href="https://github.com/mohitbhimrajka/recsys_final"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="block px-3 py-2 rounded-md text-base font-medium text-text-secondary hover:bg-surface hover:text-text-primary flex items-center gap-2 focus:outline-none focus-visible:bg-surface" // Added mobile focus
                            onClick={toggleMobileMenu}
                         >
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