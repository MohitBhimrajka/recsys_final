// frontend/src/components/Tooltip.tsx
import React, { useState, useRef, useLayoutEffect } from 'react';
import ReactDOM from 'react-dom'; // Import ReactDOM for portals
import { motion, AnimatePresence } from 'framer-motion';

interface TooltipProps {
  content: string;
  children: React.ReactElement<React.HTMLAttributes<HTMLElement>>;
  position?: 'top' | 'bottom' | 'left' | 'right';
  className?: string;
}

const Tooltip: React.FC<TooltipProps> = ({ content, children, position = 'top', className = '' }) => {
  const [isVisible, setIsVisible] = useState(false);
  const [coords, setCoords] = useState({ left: 0, top: 0 });
  const targetRef = useRef<HTMLDivElement>(null); // Ref for the wrapper div

  const showTooltip = () => setIsVisible(true);
  const hideTooltip = () => setIsVisible(false);

  // Use useLayoutEffect to calculate position *after* render but before paint
  useLayoutEffect(() => {
    if (isVisible && targetRef.current) {
      const rect = targetRef.current.getBoundingClientRect();
      let newTop = 0;
      let newLeft = 0;

      // Basic positioning logic (can be refined with tooltip dimensions)
      // Note: This requires the portal target to be the body or similar
      // We'll add scroll offsets for correct positioning relative to viewport
      const scrollX = window.scrollX;
      const scrollY = window.scrollY;

      switch (position) {
        case 'bottom':
          newTop = rect.bottom + scrollY;
          newLeft = rect.left + rect.width / 2 + scrollX;
          break;
        case 'left':
          newTop = rect.top + rect.height / 2 + scrollY;
          newLeft = rect.left + scrollX;
          break;
        case 'right':
          newTop = rect.top + rect.height / 2 + scrollY;
          newLeft = rect.right + scrollX;
          break;
        case 'top':
        default:
          newTop = rect.top + scrollY;
          newLeft = rect.left + rect.width / 2 + scrollX;
          break;
      }
      setCoords({ top: newTop, left: newLeft });
    }
  }, [isVisible, position]); // Recalculate when visibility or position changes

  const tooltipVariants = {
    hidden: { opacity: 0, y: position === 'top' ? 5 : (position === 'bottom' ? -5 : 0), scale: 0.95 },
    visible: { opacity: 1, y: 0, scale: 1, transition: { duration: 0.2, ease: 'easeOut' } },
    exit: { opacity: 0, scale: 0.95, transition: { duration: 0.15, ease: 'easeIn' } }
  };

  // Position adjustments for the tooltip itself based on coords
  let transformClasses = '';
  let offsetClasses = '';
  switch (position) {
    case 'top': transformClasses = '-translate-x-1/2'; offsetClasses = 'mb-2'; break;
    case 'bottom': transformClasses = '-translate-x-1/2'; offsetClasses = 'mt-2'; break;
    case 'left': transformClasses = '-translate-y-1/2 -translate-x-full'; offsetClasses = 'mr-2'; break; // Translate X full width left
    case 'right': transformClasses = '-translate-y-1/2'; offsetClasses = 'ml-2'; break;
  }

  const tooltipId = `tooltip-${content.replace(/\s+/g, '-').slice(0, 15)}`;

  // Tooltip JSX to be rendered in the portal
  const tooltipElement = (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          id={tooltipId}
          role="tooltip"
          className={`fixed ${offsetClasses} z-[200] w-max max-w-xs px-3 py-1.5 text-xs font-medium text-white bg-gray-900/95 dark:bg-gray-700/95 rounded-md shadow-lg backdrop-blur-sm transform ${transformClasses}`}
          style={{ left: `${coords.left}px`, top: `${coords.top}px` }} // Position using style
          variants={tooltipVariants}
          initial="hidden"
          animate="visible"
          exit="exit"
        >
          {content}
        </motion.div>
      )}
    </AnimatePresence>
  );

  return (
    // Wrapper div to attach events and ref
    <div
      ref={targetRef}
      className={`relative inline-flex ${className}`}
      onMouseEnter={showTooltip}
      onMouseLeave={hideTooltip}
      onFocus={showTooltip}
      onBlur={hideTooltip}
    >
      {React.cloneElement(children, { 'aria-describedby': tooltipId })}
      {/* Render the tooltip into the body using a portal */}
      {typeof document !== 'undefined' ? ReactDOM.createPortal(tooltipElement, document.body) : null}
    </div>
  );
};

export default Tooltip;