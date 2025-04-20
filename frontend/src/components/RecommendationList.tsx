// frontend/src/components/RecommendationList.tsx
import React from 'react';
import { RecommendationItem } from '../types'; // Adjust path if needed
import RecommendationCard from './RecommendationCard';
import { motion, AnimatePresence } from 'framer-motion'; // Import framer-motion

interface RecommendationListProps {
  recommendations: RecommendationItem[];
  selectedUserId: number | null; // Receive selectedUserId to display in message
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
      stiffness: 100,
      damping: 12
    }
  }
};

const RecommendationList: React.FC<RecommendationListProps> = ({ recommendations, selectedUserId }) => {

  // Message when user is selected but fetch returns no recommendations
  if (selectedUserId && recommendations.length === 0) {
    return (
      <div className="text-center mt-10 p-6 bg-yellow-50 border border-yellow-300 rounded-md shadow-sm max-w-md mx-auto">
        <p className="text-yellow-800 font-medium">
            No recommendations found for student ID <span className="font-bold">{selectedUserId}</span>.
        </p>
        <p className="text-sm text-yellow-700 mt-1">
            This user might be new, may not have enough interaction data after filtering, or may have already interacted with all relevant courses.
        </p>
      </div>
      );
  }

  // Don't render anything if no user is selected (handled by parent)
  // or if recommendations are empty (handled above)
  if (!selectedUserId || recommendations.length === 0) {
      return null;
  }


  return (
    <div>
      <h2 className="text-xl font-semibold mb-6 text-center">
          Top {recommendations.length} Recommendations for Student <span className='font-bold text-indigo-600'>{selectedUserId}</span>
      </h2>
      {/* AnimatePresence helps animate items entering/leaving, though not strictly needed for just entry */}
      <AnimatePresence>
          <motion.div
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6" // Increased gap slightly
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            exit="hidden" // Define exit animation if needed
          >
            {recommendations.map((rec, index) => (
              <motion.div key={rec.presentation_id} variants={itemVariants}> {/* Key moved here */}
                  <RecommendationCard recommendation={rec} rank={index + 1} />
              </motion.div>
            ))}
          </motion.div>
      </AnimatePresence>
    </div>
  );
};

export default RecommendationList;