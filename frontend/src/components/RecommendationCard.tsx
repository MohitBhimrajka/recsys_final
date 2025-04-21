import React from 'react';
import { RecommendationItem } from '../types';
import { motion } from 'framer-motion';
import Tooltip from './Tooltip';
import { FiInfo, FiEyeOff, FiStar, FiZoomIn } from 'react-icons/fi'; // Removed FiExternalLink as ZoomIn covers detail view

interface RecommendationCardProps {
  recommendation: RecommendationItem;
  rank: number;
  isHidden: boolean;
  isHighlighted: boolean;
  onHide: (presentationId: string) => void;
  onHighlight: (presentationId: string) => void;
  onClick: (recommendation: RecommendationItem) => void; // For detail modal
  isIndividualModelCard?: boolean;
  modelName?: string;
}

// Subtle hover effect variant
const cardHoverVariant = {
  hover: {
    scale: 1.03,
    boxShadow: '0 10px 15px -3px rgba(var(--color-primary-default-rgb), 0.1), 0 4px 6px -4px rgba(var(--color-primary-default-rgb), 0.1)', // Use primary color shadow
    transition: { duration: 0.25, ease: [0.4, 0, 0.2, 1] } // Smoother ease
  }
};

// Variant for the rank badge entrance
const rankBadgeVariant = {
    hidden: { scale: 0.5, opacity: 0 },
    visible: { scale: 1, opacity: 1, transition: { type: 'spring', stiffness: 300, damping: 15, delay: 0.1 } }
};

const RecommendationCard: React.FC<RecommendationCardProps> = ({
    recommendation, rank, isHidden, isHighlighted, onHide, onHighlight, onClick,
    isIndividualModelCard = false, modelName
}) => {
  // Base styling with minimum height to prevent collapse
  const cardBaseClasses = "bg-surface rounded-lg shadow-lg overflow-hidden border h-full flex flex-col justify-between transition-all duration-300 ease-out relative group min-h-[200px]"; // Increased min-h slightly
  // Dynamic border and shadow based on state
  const borderClasses = isHighlighted
    ? "border-primary shadow-primary/20 ring-2 ring-primary/50 ring-offset-2 ring-offset-surface" // Prominent highlight with ring
    : "border-border-color hover:border-primary/60"; // Slightly stronger hover border
  // Opacity and grayscale for hidden state
  const opacityClass = isHidden ? "opacity-40 grayscale" : "opacity-100";
  // Cursor style based on hidden state
  const cursorClass = isHidden ? "cursor-default" : "cursor-pointer";

  // Click handlers with propagation stop
  const handleHideClick = (e: React.MouseEvent) => { e.stopPropagation(); onHide(recommendation.presentation_id); };
  const handleHighlightClick = (e: React.MouseEvent) => { e.stopPropagation(); onHighlight(recommendation.presentation_id); };
  const handleCardClick = () => { if (!isHidden) onClick(recommendation); };

  // Dynamic tooltip content for score
  const scoreTooltipContent = isIndividualModelCard
   ? `Predicted score from ${modelName || 'this model'}. Higher is generally better, but rank matters most.`
   : "Combined ensemble score (weighted average). Higher is generally better.";

  return (
    <motion.div
        className={`${cardBaseClasses} ${borderClasses} ${opacityClass} ${cursorClass}`}
        layout // Enable smooth layout transitions if list reorders
        variants={cardHoverVariant}
        whileHover={!isHidden ? "hover" : ""} // Apply hover only if not hidden
        onClick={handleCardClick}
        title={!isHidden ? `View details for ${recommendation.presentation_id}` : "Recommendation hidden"}
        aria-hidden={isHidden} // Hide from accessibility tree if visually hidden
    >
      {/* Rank Badge - Animated Entrance */}
      <motion.div
         className={`absolute top-3 left-3 z-10 text-xs font-bold uppercase tracking-wider text-background px-2.5 py-0.5 rounded-full shadow ${isHighlighted ? 'bg-yellow-400' : 'bg-primary'}`}
         variants={rankBadgeVariant}
         initial="hidden"
         animate="visible"
      >
          Rank #{rank}
      </motion.div>

      {/* Interactive Buttons Overlay (Only for Ensemble cards) */}
      {!isHidden && !isIndividualModelCard && (
         <div className="absolute top-2 right-2 z-20 flex space-x-1 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity duration-200"> {/* Also show on focus-within */}
             {/* Highlight Button */}
             <Tooltip content={isHighlighted ? "Remove Highlight" : "Highlight"} position="top">
                 <motion.button whileTap={{ scale: 0.9 }} onClick={handleHighlightClick} className={`p-1.5 rounded-full transition-colors ${isHighlighted ? 'bg-yellow-600/30 text-yellow-300 hover:bg-yellow-600/40' : 'bg-surface/80 text-text-muted hover:text-yellow-400 hover:bg-border-color'}`} aria-label={isHighlighted ? "Remove highlight" : "Highlight"}>
                     <FiStar size={14} />
                 </motion.button>
             </Tooltip>
              {/* Hide Button */}
             <Tooltip content="Hide" position="top">
                 <motion.button whileTap={{ scale: 0.9 }} onClick={handleHideClick} className="p-1.5 rounded-full bg-surface/80 text-text-muted hover:text-red-400 hover:bg-border-color transition-colors" aria-label="Hide">
                     <FiEyeOff size={14} />
                 </motion.button>
             </Tooltip>
              {/* View Details Button */}
              <Tooltip content="View Details" position="top">
                 <motion.button whileTap={{ scale: 0.9 }} onClick={(e) => { e.stopPropagation(); handleCardClick(); }} className="p-1.5 rounded-full bg-surface/80 text-text-muted hover:text-primary hover:bg-border-color transition-colors" aria-label="View details">
                     <FiZoomIn size={14} />
                 </motion.button>
             </Tooltip>
         </div>
      )}

      {/* Main Content Area */}
      <div className="p-5 pt-10 flex flex-col flex-grow"> {/* pt-10 creates space for rank badge */}

        {/* Presentation ID */}
        <h3 className="text-base sm:text-lg font-semibold text-text-primary mb-2 truncate leading-tight" title={recommendation.presentation_id}>
            {recommendation.presentation_id}
        </h3>

        {/* Details Section */}
        <div className="text-xs sm:text-sm text-text-muted space-y-1 mb-4 flex-grow"> {/* Adjusted text size */}
            <p><span className="font-medium text-text-secondary/90">Module:</span> {recommendation.module_id}</p>
            <p><span className="font-medium text-text-secondary/90">Code:</span> {recommendation.presentation_code}</p>
            {recommendation.module_presentation_length != null && (
                 <p><span className="font-medium text-text-secondary/90">Duration:</span> {recommendation.module_presentation_length} days</p>
            )}
        </div>

         {/* Score Section at the Bottom */}
         <div className='mt-auto border-t border-border-color/50 pt-3'>
            <div className="flex justify-between items-center text-sm"> {/* Removed font-semibold from label */}
                 <span className="text-text-secondary/90">Predicted Score</span>
                 <Tooltip content={scoreTooltipContent} position="top" delay={100}>
                     <span className="text-primary font-bold flex items-center gap-1 cursor-help text-base md:text-lg"> {/* Larger score */}
                         {recommendation.score.toFixed(4)} {/* 4 decimal places */}
                         <FiInfo size={12} className="opacity-70" />
                     </span>
                 </Tooltip>
             </div>
         </div>
      </div>
    </motion.div>
  );
};

export default RecommendationCard;