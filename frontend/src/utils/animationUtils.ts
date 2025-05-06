// frontend/src/utils/animationUtils.ts
// Utility functions and constants for animations across the application

/**
 * Optimized transition settings to reduce jitter and provide smoother animations
 * These use hardware acceleration and optimized easing
 */
export const smoothTransition = {
  type: 'tween',
  ease: [0.25, 0.1, 0.25, 1], // cubic-bezier for smooth natural movement
  duration: 0.3
};

/**
 * Common animation variants optimized for performance
 * Uses transform-based animations rather than layout properties where possible
 */
export const optimizedVariants = {
  // Standard fade + slide variants
  hidden: { 
    opacity: 0, 
    y: 10, 
    transition: smoothTransition 
  },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: smoothTransition
  },
  
  // Hardware accelerated variants that reduce jitter
  hardwareAccelerated: {
    initial: { 
      opacity: 0, 
      transform: 'translate3d(0, 10px, 0)' 
    },
    animate: { 
      opacity: 1, 
      transform: 'translate3d(0, 0, 0)',
      transition: {
        ...smoothTransition,
        willChange: 'opacity, transform'
      }
    }
  },
  
  // Variants for components that should appear without moving
  fade: {
    hidden: { opacity: 0 },
    visible: { 
      opacity: 1,
      transition: {
        duration: 0.2
      }
    }
  },
  
  // Container variants that stagger children
  container: {
    hidden: { opacity: 0 },
    visible: { 
      opacity: 1,
      transition: {
        staggerChildren: 0.05,
        when: 'beforeChildren'
      }
    }
  },
  
  // Expand/collapse for tree nodes and accordion components
  expand: {
    collapsed: { 
      height: 0, 
      opacity: 0,
      overflow: 'hidden'
    },
    expanded: { 
      height: 'auto', 
      opacity: 1,
      overflow: 'visible',
      transition: {
        height: {
          type: 'spring',
          stiffness: 400,
          damping: 30
        },
        opacity: {
          duration: 0.2
        }
      }
    }
  }
};

/**
 * Apply hardware acceleration class properties to reduce jitter
 * Can be spread into component style objects
 */
export const hardwareAcceleration = {
  willChange: 'transform',
  transform: 'translateZ(0)'
}; 