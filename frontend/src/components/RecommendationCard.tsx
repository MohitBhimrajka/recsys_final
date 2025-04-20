// frontend/src/components/RecommendationCard.tsx
import React from 'react';
import { RecommendationItem } from '../types';
import { motion } from 'framer-motion';
import Tooltip from './Tooltip';
import { FiInfo, FiEyeOff, FiStar, FiExternalLink } from 'react-icons/fi'; // Added icons

interface RecommendationCardProps {
  recommendation: RecommendationItem;
  rank: number;
  isHidden: boolean;
  isHighlighted: boolean;
  onHide: (presentationId: string) => void;
  onHighlight: (presentationId: string) => void;
  onClick: (recommendation: RecommendationItem) => void; // Added onClick handler
}

const cardHoverVariant = {
  hover: {
    scale: 1.02, // Slightly less aggressive scale
    // Gradient border applied via Tailwind class manipulation below
    transition: { duration: 0.2, ease: 'easeOut' }
  }
};

// Simple visual bar for score
const ScoreBar: React.FC<{ score: number }> = ({ score }) => {
  // Clamp score between 0 and a reasonable max (e.g., 1.0 or slightly higher if scores can exceed)
  const clampedScore = Math.max(0, Math.min(1.0, score / 1.5)); // Adjust divisor based on typical score range
  const widthPercent = `${Math.round(clampedScore * 100)}%`;
  return (
    <div className="w-full bg-gray-700 rounded-full h-1.5 dark:bg-border-color mt-1">
      <div
        className="bg-primary h-1.5 rounded-full transition-all duration-300 ease-out"
        style={{ width: widthPercent }}
      ></div>
    </div>
  );
};


const RecommendationCard: React.FC<RecommendationCardProps> = ({
    recommendation, rank, isHidden, isHighlighted, onHide, onHighlight, onClick
}) => {
  const cardBaseClasses = "bg-surface rounded-lg shadow-lg overflow-hidden border h-full flex flex-col justify-between transition-all duration-300 ease-out";
  const borderClasses = isHighlighted
    ? "border-primary shadow-primary/20"
    : "border-border-color hover:border-primary/40";
  const opacityClass = isHidden ? "opacity-40 grayscale" : "opacity-100";

  const handleHideClick = (e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent card click
    onHide(recommendation.presentation_id);
  };

  const handleHighlightClick = (e: React.MouseEvent) => {
     e.stopPropagation(); // Prevent card click
     onHighlight(recommendation.presentation_id);
  };

  const handleCardClick = () => {
      onClick(recommendation);
  }

  return (
    <motion.div
        className={`${cardBaseClasses} ${borderClasses} ${opacityClass} group cursor-pointer relative`}
        layout // Keep layout for smooth list animations
        variants={cardHoverVariant}
        whileHover="hover"
        onClick={handleCardClick}
    >
      {/* Interactive Buttons Overlay - Visible on Hover */}
      {!isHidden && (
         <div className="absolute top-2 right-2 z-10 flex space-x-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
             <Tooltip content={isHighlighted ? "Remove Highlight" : "Highlight"} position="top">
                 <button
                     onClick={handleHighlightClick}
                     className={`p-1.5 rounded-full transition-colors ${isHighlighted ? 'bg-yellow-500/20 text-yellow-400 hover:bg-yellow-500/30' : 'bg-surface/80 text-text-muted hover:text-yellow-400 hover:bg-border-color'}`}
                     aria-label={isHighlighted ? "Remove highlight from this recommendation" : "Highlight this recommendation"}
                 >
                     <FiStar size={14} />
                 </button>
             </Tooltip>
             <Tooltip content="Hide" position="top">
                 <button
                     onClick={handleHideClick}
                     className="p-1.5 rounded-full bg-surface/80 text-text-muted hover:text-red-400 hover:bg-border-color transition-colors"
                     aria-label="Hide this recommendation"
                 >
                     <FiEyeOff size={14} />
                 </button>
             </Tooltip>
         </div>
      )}

      <div className="p-5 flex flex-col flex-grow"> {/* Allow content to grow */}
         {/* Header with Rank and Score */}
         <div className="flex justify-between items-start mb-3"> {/* Use items-start */}
            <span className={`text-xs font-bold uppercase tracking-wider text-background px-2.5 py-0.5 rounded-full ${isHighlighted ? 'bg-yellow-400' : 'bg-primary'}`}>
                Rank #{rank}
            </span>
             {/* Score moved below title */}
        </div>

        {/* Main Content */}
        <h3 className="text-lg font-semibold text-text-primary mb-2 truncate" title={recommendation.presentation_id}>
            {recommendation.presentation_id}
        </h3>
        <div className="text-sm text-text-muted space-y-1 mb-4 flex-grow"> {/* Add flex-grow */}
            <p><span className="font-medium text-text-secondary">Module:</span> {recommendation.module_id}</p>
            <p><span className="font-medium text-text-secondary">Presentation:</span> {recommendation.presentation_code}</p>
            {recommendation.module_presentation_length != null && (
                 <p><span className="font-medium text-text-secondary">Duration:</span> {recommendation.module_presentation_length} days</p>
            )}
        </div>

         {/* Score section at the bottom */}
         <div className='mt-auto'> {/* Pushes score to bottom */}
            <div className="flex justify-between items-center text-sm font-semibold text-text-secondary">
                 <span>Predicted Score</span>
                 <Tooltip content="Model's predicted relevance score (higher is better)" position="top">
                     <span className="text-primary font-bold flex items-center gap-1 cursor-help">
                         {recommendation.score.toFixed(3)}
                         <FiInfo size={12} className="opacity-60" />
                     </span>
                 </Tooltip>
             </div>
             <ScoreBar score={recommendation.score} />
         </div>

      </div>
      {/* Optional: Add a subtle external link icon on hover */}
      <FiExternalLink size={14} className="absolute bottom-3 right-3 text-text-muted opacity-0 group-hover:opacity-50 transition-opacity duration-200" />
    </motion.div>
  );
};

export default RecommendationCard;