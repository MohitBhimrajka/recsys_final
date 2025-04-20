// frontend/src/components/RecommendationList.tsx
import React from 'react';
import { RecommendationItem } from '../types';
import RecommendationCard from './RecommendationCard';
import { motion, AnimatePresence } from 'framer-motion';

interface RecommendationListProps {
  recommendations: RecommendationItem[];
  selectedUserId: number | null;
  // New props for card interactions
  hiddenCards: Set<string>;
  highlightedCards: Set<string>;
  onHideCard: (presentationId: string) => void;
  onHighlightCard: (presentationId: string) => void;
  onCardClick: (recommendation: RecommendationItem) => void;
}

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.07, delayChildren: 0.1 } // Slightly faster stagger
  }
};

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: { y: 0, opacity: 1, transition: { type: "spring", stiffness: 100, damping: 15 } }, // Adjusted spring
  exit: { opacity: 0, y: -10, transition: { duration: 0.2 } }
};

const RecommendationList: React.FC<RecommendationListProps> = ({
    recommendations, selectedUserId,
    hiddenCards, highlightedCards, onHideCard, onHighlightCard, onCardClick
 }) => {

  // Filter out hidden recommendations before mapping
  const visibleRecommendations = recommendations.filter(rec => !hiddenCards.has(rec.presentation_id));

  if (selectedUserId && visibleRecommendations.length === 0 && recommendations.length > 0) {
     // Case where all recommendations are hidden by the user
     return (
       <div className="text-center mt-12 p-8 bg-surface border border-border-color rounded-lg shadow-md max-w-lg mx-auto">
         <div className="text-primary text-5xl mb-4">🙈</div>
         <p className="text-text-primary font-medium text-lg mb-2"> All Recommendations Hidden </p>
         <p className="text-sm text-text-muted">
             You've hidden all the current suggestions for student ID <span className="font-bold text-text-secondary">{selectedUserId}</span>. Select a new student or unhide cards (functionality to unhide not implemented in this version).
         </p>
       </div>
     );
   }

  if (selectedUserId && recommendations.length === 0) {
    // Original empty state - no recommendations returned from API
    return (
      <div className="text-center mt-12 p-8 bg-surface border border-border-color rounded-lg shadow-md max_w-lg mx-auto">
        <div className="text-primary text-5xl mb-4">🤔</div>
        <p className="text-text-primary font-medium text-lg mb-2"> No Recommendations Found </p>
        <p className="text-sm text-text-muted">
            We couldn't find any suitable course recommendations for student ID <span className="font-bold text-text-secondary">{selectedUserId}</span>.
            This might happen if the student is new, hasn't interacted enough, or has already seen most relevant courses.
        </p>
      </div>
      );
  }

  if (!selectedUserId || recommendations.length === 0) { return null; }

  return (
    <div>
      <AnimatePresence>
          <motion.div
            className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-6 md:gap-8"
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            exit="hidden"
          >
            {/* Map over the original recommendations to preserve index for rank, but check visibility */}
            {recommendations.map((rec, index) => (
              <motion.div
                  key={rec.presentation_id}
                  variants={itemVariants}
                  layout
                  // Animate presence/absence based on hidden state
                  initial={false} // Don't run initial animation on filter change
                  animate={hiddenCards.has(rec.presentation_id) ? "hidden" : "visible"}
                  exit="exit" // Use exit animation when hiding
              >
                  <RecommendationCard
                      recommendation={rec}
                      rank={index + 1}
                      isHidden={hiddenCards.has(rec.presentation_id)}
                      isHighlighted={highlightedCards.has(rec.presentation_id)}
                      onHide={onHideCard}
                      onHighlight={onHighlightCard}
                      onClick={onCardClick}
                  />
              </motion.div>
            ))}
          </motion.div>
      </AnimatePresence>
    </div>
  );
};

export default RecommendationList;