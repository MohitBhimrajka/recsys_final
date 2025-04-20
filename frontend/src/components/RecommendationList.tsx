// frontend/src/components/RecommendationList.tsx
import React from 'react';
import { RecommendationItem } from '../types';
import RecommendationCard from './RecommendationCard';
import { motion, AnimatePresence } from 'framer-motion';

interface RecommendationListProps {
  recommendations: RecommendationItem[];
  selectedUserId: number | null;
}

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.08, delayChildren: 0.1, }
  }
};

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: { y: 0, opacity: 1, transition: { type: "spring", stiffness: 120, damping: 14 } },
  exit: { opacity: 0, y: -10, transition: { duration: 0.2 } }
};

const RecommendationList: React.FC<RecommendationListProps> = ({ recommendations, selectedUserId }) => {

  if (selectedUserId && recommendations.length === 0) {
    return (
      <div className="text-center mt-12 p-8 bg-surface border border-border-color rounded-lg shadow-md max-w-lg mx-auto">
        <div className="text-primary text-5xl mb-4">🤔</div>
        <p className="text-text-primary font-medium text-lg mb-2"> No Recommendations Found </p>
        <p className="text-sm text-text-muted">
            We couldn't find any suitable course recommendations for student ID <span className="font-bold text-text-secondary">{selectedUserId}</span> at this time.
            This might happen if the student is new, hasn't interacted enough, or has already seen most relevant courses.
        </p>
      </div>
      );
  }

  if (!selectedUserId || recommendations.length === 0) { return null; }

  return (
    <div>
      {/* Title now handled in DemoPage */}
      <AnimatePresence>
          <motion.div
            className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-6 md:gap-8" // Use xl:grid-cols-3 for wider screens
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            exit="hidden"
          >
            {recommendations.map((rec, index) => (
              <motion.div key={rec.presentation_id} variants={itemVariants} layout >
                  <RecommendationCard recommendation={rec} rank={index + 1} />
              </motion.div>
            ))}
          </motion.div>
      </AnimatePresence>
    </div>
  );
};

export default RecommendationList;