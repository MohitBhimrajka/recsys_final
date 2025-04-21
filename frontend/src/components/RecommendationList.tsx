// frontend/src/components/RecommendationList.tsx
import React from 'react';
import { RecommendationItem } from '../types';
import RecommendationCard from './RecommendationCard';
import { motion, AnimatePresence } from 'framer-motion';
import { FiInbox, FiEyeOff } from "react-icons/fi"; // Added icons

interface RecommendationListProps {
  recommendations: RecommendationItem[];
  selectedUserId: number | null;
  // Props for card interactions (primarily for Ensemble list)
  hiddenCards?: Set<string>;
  highlightedCards?: Set<string>;
  onHideCard?: (presentationId: string) => void;
  onHighlightCard?: (presentationId: string) => void;
  onCardClick: (recommendation: RecommendationItem) => void; // Now mandatory for detail modal
  // Optional props for individual model lists
  isIndividualModelList?: boolean;
  modelName?: string;
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
  exit: { opacity: 0, scale: 0.95, transition: { duration: 0.2 } } // Add scale on exit
};

const RecommendationList: React.FC<RecommendationListProps> = ({
    recommendations, selectedUserId,
    hiddenCards = new Set(), // Default to empty sets if not provided
    highlightedCards = new Set(),
    onHideCard = () => {}, // Default to no-op functions
    onHighlightCard = () => {},
    onCardClick, // Mandatory
    isIndividualModelList = false,
    modelName
 }) => {

  // Filter out hidden recommendations *only* if it's the ensemble list
  const displayRecommendations = isIndividualModelList
    ? recommendations
    : recommendations.filter(rec => !hiddenCards.has(rec.presentation_id));

  // --- Empty State: All Recommendations Hidden (Only for Ensemble) ---
  if (!isIndividualModelList && selectedUserId && displayRecommendations.length === 0 && recommendations.length > 0) {
     return (
       <motion.div
           initial={{ opacity: 0, y: 10 }}
           animate={{ opacity: 1, y: 0 }}
           className="text-center mt-12 p-8 bg-surface/50 border border-border-color rounded-lg shadow-md max-w-lg mx-auto"
       >
         <FiEyeOff className="text-primary text-5xl mb-4 mx-auto opacity-60" /> {/* Changed icon */}
         <p className="text-text-primary font-semibold text-lg mb-2"> All Suggestions Hidden </p> {/* Changed wording */}
         <p className="text-sm text-text-muted">
             You've hidden all the combined suggestions for student <span className="font-bold text-text-secondary">{selectedUserId}</span>.
             {/* Placeholder for future unhide feature */}
             {/* <button className="text-primary hover:underline text-sm mt-2">Unhide all?</button> */}
         </p>
       </motion.div>
     );
   }

  // --- Empty State: No Recommendations Found ---
  if (selectedUserId && recommendations.length === 0) {
    const message = isIndividualModelList
       ? `Model "${modelName}" couldn't find suitable recommendations for student ${selectedUserId}.`
       : `We couldn't find any combined recommendations for student ${selectedUserId}.`;
    return (
      <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mt-12 p-8 bg-surface/50 border border-border-color rounded-lg shadow-md max-w-lg mx-auto"
      >
        <FiInbox className="text-primary text-5xl mb-4 mx-auto opacity-60" /> {/* Changed icon */}
        <p className="text-text-primary font-semibold text-lg mb-2"> No Recommendations Found </p> {/* Consistent title */}
        <p className="text-sm text-text-muted">
            {message} This might happen if the student is new, hasn't interacted enough, or has already seen most relevant courses.
        </p>
      </motion.div>
      );
  }

  // --- Default: No User Selected or Empty Initial Recommendations ---
  // Render nothing in this case, DemoPage handles the initial prompt.
  if (!selectedUserId || recommendations.length === 0) { return null; }

  // --- Render List ---
  return (
    <div>
      <AnimatePresence> {/* Enable exit animations */}
          <motion.div
            className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-6 md:gap-8"
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            exit="hidden" // Define exit variant for the container if needed
          >
            {/* Map over the original recommendations to preserve index for rank, but check visibility for ensemble */}
            {recommendations.map((rec, index) => {
               const isHidden = !isIndividualModelList && hiddenCards.has(rec.presentation_id);
               // Conditionally render based on hidden state ONLY for ensemble
               return (
                   <motion.div
                       key={rec.presentation_id} // Use item ID as key
                       variants={itemVariants}
                       layout // Enable layout animation
                       initial="hidden" // Animate entrance
                       animate="visible"
                       exit="exit" // Apply exit animation when hidden
                       // Use AnimatePresence's visibility or manually control animation based on isHidden
                       // style={{ display: isHidden ? 'none' : 'block' }} // Simple hide/show
                       // OR better with AnimatePresence: wrap the card itself if you want exit animations
                   >
                     {!isHidden && ( // Render card only if not hidden (AnimatePresence handles the exit)
                       <RecommendationCard
                           recommendation={rec}
                           rank={index + 1}
                           isHidden={isHidden}
                           isHighlighted={highlightedCards.has(rec.presentation_id)}
                           onHide={onHideCard}
                           onHighlight={onHighlightCard}
                           onClick={onCardClick} // Pass mandatory click handler
                           // Pass down identifier props
                           isIndividualModelCard={isIndividualModelList}
                           modelName={modelName}
                       />
                     )}
                   </motion.div>
               );
            })}
          </motion.div>
      </AnimatePresence>
    </div>
  );
};

export default RecommendationList;