// frontend/src/components/RecommendationList.tsx
import React from 'react';
import { RecommendationItem } from '../types';
import RecommendationCard from './RecommendationCard';
import { motion, AnimatePresence } from 'framer-motion';

interface RecommendationListProps {
  recommendations: RecommendationItem[];
  selectedUserId: number | null;
}

// Animation variants for the container and list items
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.08, // Stagger the animation of children
      delayChildren: 0.1,
    }
  }
};

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
    transition: {
      type: "spring",
      stiffness: 120, // Slightly stiffer spring
      damping: 14
    }
  },
   exit: { // Define exit animation if needed when list changes
    opacity: 0,
    y: -10,
    transition: { duration: 0.2 }
  }
};

const RecommendationList: React.FC<RecommendationListProps> = ({ recommendations, selectedUserId }) => {

  // Message when user is selected but fetch returns no recommendations
  if (selectedUserId && recommendations.length === 0) {
    return (
      // Restyled "No Recommendations" message
      <div className="text-center mt-12 p-8 bg-surface border border-border-color rounded-lg shadow-md max-w-lg mx-auto">
        <div className="text-primary text-5xl mb-4">🤔</div> {/* Simple Emoji Icon */}
        <p className="text-text-primary font-medium text-lg mb-2">
            No Recommendations Found
        </p>
        <p className="text-sm text-text-muted">
            We couldn't find any suitable course recommendations for student ID <span className="font-bold text-text-secondary">{selectedUserId}</span> at this time.
            This might happen if the student is new, hasn't interacted enough, or has already seen most relevant courses.
        </p>
      </div>
      );
  }

  // Should not render if no user selected (handled by parent) or handled above
  if (!selectedUserId || recommendations.length === 0) {
      return null;
  }

  return (
    <div>
      <h2 className="text-2xl font-semibold mb-8 text-center text-text-primary">
          Top Recommendations for Student <span className='font-bold text-primary'>{selectedUserId}</span>
      </h2>
      {/* Use AnimatePresence for smooth updates if recommendations change */}
      <AnimatePresence>
          <motion.div
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8" // Adjust grid gap
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            exit="hidden" // Animate container exit
          >
            {recommendations.map((rec, index) => (
              // Apply item variant to each card wrapper
              <motion.div
                key={rec.presentation_id} // Key must be stable and unique
                variants={itemVariants}
                layout // Enable smooth layout changes
               >
                  <RecommendationCard recommendation={rec} rank={index + 1} />
              </motion.div>
            ))}
          </motion.div>
      </AnimatePresence>
    </div>
  );
};

export default RecommendationList;