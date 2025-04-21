// frontend/src/components/Tooltip.tsx
import React, { useState, useRef, useLayoutEffect } from 'react';
import ReactDOM from 'react-dom'; // Import ReactDOM for portals
import { motion, AnimatePresence } from 'framer-motion';

interface TooltipProps {
  content: React.ReactNode; // Allow React nodes for content
  children: React.ReactElement<React.HTMLAttributes<HTMLElement>>;
  position?: 'top' | 'bottom' | 'left' | 'right';
  className?: string;
  delay?: number; // Optional delay before showing
}

const Tooltip: React.FC<TooltipProps> = ({ content, children, position = 'top', className = '', delay = 0 }) => {
  const [isVisible, setIsVisible] = useState(false);
  const [coords, setCoords] = useState({ left: 0, top: 0 });
  const targetRef = useRef<HTMLDivElement>(null); // Ref for the wrapper div
  const timeoutRef = useRef<number | null>(null); // Ref for delay timeout

  const showTooltip = () => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current); // Clear any existing hide timeout
    timeoutRef.current = setTimeout(() => {
        setIsVisible(true);
    }, delay);
  };

  const hideTooltip = () => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current); // Clear any existing show timeout
    setIsVisible(false);
  };

  // Use useLayoutEffect to calculate position *after* render but before paint
  useLayoutEffect(() => {
    if (isVisible && targetRef.current) {
      const targetRect = targetRef.current.getBoundingClientRect();
      // Estimate tooltip dimensions (adjust if needed, or measure after first render)
      const tooltipHeightEstimate = 30;
      const tooltipWidthEstimate = 100; // Very rough estimate
      const gap = 8; // Space between target and tooltip

      let newTop = 0;
      let newLeft = 0;

      // Positioning logic relative to viewport
      const scrollX = window.scrollX;
      const scrollY = window.scrollY;

      switch (position) {
        case 'bottom':
          newTop = targetRect.bottom + scrollY + gap;
          newLeft = targetRect.left + scrollX + targetRect.width / 2;
          break;
        case 'left':
          newTop = targetRect.top + scrollY + targetRect.height / 2;
          newLeft = targetRect.left + scrollX - gap; // Move left of target
          break;
        case 'right':
          newTop = targetRect.top + scrollY + targetRect.height / 2;
          newLeft = targetRect.right + scrollX + gap; // Move right of target
          break;
        case 'top':
        default:
          newTop = targetRect.top + scrollY - tooltipHeightEstimate - gap; // Position above target
          newLeft = targetRect.left + scrollX + targetRect.width / 2; // Center horizontally
          break;
      }
      setCoords({ top: newTop, left: newLeft });
    }
  }, [isVisible, position]); // Recalculate when visibility or position changes

  // Framer motion variants
  const tooltipVariants = {
    hidden: { opacity: 0, y: position === 'top' ? 5 : (position === 'bottom' ? -5 : 0), scale: 0.95 },
    visible: { opacity: 1, y: 0, scale: 1, transition: { duration: 0.2, ease: 'easeOut' } },
    exit: { opacity: 0, scale: 0.95, transition: { duration: 0.15, ease: 'easeIn' } }
  };

  // Transform classes based on position to center the tooltip pointer
  let transformClasses = '';
  switch (position) {
    case 'top': transformClasses = '-translate-x-1/2'; break; // Center horizontally
    case 'bottom': transformClasses = '-translate-x-1/2'; break; // Center horizontally
    case 'left': transformClasses = '-translate-y-1/2 -translate-x-full'; break; // Translate full width left, center vertically
    case 'right': transformClasses = '-translate-y-1/2'; break; // Center vertically
  }

  const tooltipId = `tooltip-${Math.random().toString(36).substring(7)}`; // Generate somewhat unique ID

  // Tooltip JSX to be rendered in the portal
  const tooltipElement = (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          id={tooltipId}
          role="tooltip"
          className={`fixed z-[200] w-max max-w-xs px-3 py-1.5 text-xs font-medium text-white bg-gray-900/95 dark:bg-gray-800/95 rounded-md shadow-lg backdrop-blur-sm transform ${transformClasses}`} // Adjusted colors slightly
          style={{ left: `${coords.left}px`, top: `${coords.top}px` }} // Position using style
          variants={tooltipVariants}
          initial="hidden"
          animate="visible"
          exit="exit"
          // Prevent tooltip from stealing focus
          aria-hidden="true"
        >
          {content}
        </motion.div>
      )}
    </AnimatePresence>
  );

  // Check if running in a browser environment before creating portal
   const portalTarget = typeof document !== 'undefined' ? document.body : null;

  return (
    // Wrapper div to attach events and ref
    <div
      ref={targetRef}
      className={`relative inline-flex ${className}`}
      onMouseEnter={showTooltip}
      onMouseLeave={hideTooltip}
      onFocus={showTooltip} // Show on focus for accessibility
      onBlur={hideTooltip}  // Hide on blur for accessibility
    >
      {/* Clone child to add aria-describedby for accessibility */}
      {React.cloneElement(children, { 'aria-describedby': tooltipId })}
      {/* Render the tooltip into the body using a portal if possible */}
      {portalTarget ? ReactDOM.createPortal(tooltipElement, portalTarget) : null}
    </div>
  );
};

export default Tooltip;