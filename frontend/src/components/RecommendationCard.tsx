// frontend/src/components/RecommendationCard.tsx
import React from 'react';
import { RecommendationItem } from '../types';
import { motion } from 'framer-motion'; // Import motion

interface RecommendationCardProps {
  recommendation: RecommendationItem;
  rank: number;
}

// Animation variant for individual card hover effect (optional)
const cardHoverVariant = {
  hover: {
    scale: 1.03,
    boxShadow: "0px 10px 20px rgba(0, 0, 0, 0.2)", // Darker shadow on hover
    transition: { duration: 0.2 }
  }
};

const RecommendationCard: React.FC<RecommendationCardProps> = ({ recommendation, rank }) => {
  return (
    // Apply motion for hover and list animations (variants provided by list)
    <motion.div
        className="bg-surface rounded-lg shadow-lg overflow-hidden border border-border-color h-full flex flex-col justify-between" // Ensure card takes full height of grid cell
        layout // Add layout prop for smoother animations if size changes
        variants={cardHoverVariant}
        whileHover="hover"
    >
      <div className="p-5">
         {/* Header with Rank and Score */}
         <div className="flex justify-between items-center mb-4 pb-2 border-b border-border-color">
            <span className="text-xs font-bold uppercase tracking-wider text-background bg-primary px-2.5 py-1 rounded-full">
                Rank #{rank}
            </span>
            <span className="text-sm font-semibold text-text-secondary">
                Score: <span className="text-primary font-bold">{recommendation.score.toFixed(3)}</span>
             </span>
        </div>

        {/* Main Content */}
        <h3 className="text-lg font-semibold text-text-primary mb-2 truncate" title={recommendation.presentation_id}>
            {recommendation.presentation_id}
        </h3>
        <div className="text-sm text-text-muted space-y-1">
            <p>
                <span className="font-medium text-text-secondary">Module:</span> {recommendation.module_id}
            </p>
            <p>
                 <span className="font-medium text-text-secondary">Presentation:</span> {recommendation.presentation_code}
            </p>
            {recommendation.module_presentation_length != null && (
                 <p>
                     <span className="font-medium text-text-secondary">Duration:</span> {recommendation.module_presentation_length} days
                </p>
            )}
        </div>
      </div>

      {/* Optional Footer - can add links or actions here */}
      {/* <div className="bg-background px-5 py-3 mt-auto border-t border-border-color">
        <button className="text-sm text-primary hover:text-primary-dark font-medium">
          View Details →
        </button>
      </div> */}
    </motion.div>
  );
};

export default RecommendationCard;