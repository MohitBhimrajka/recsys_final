// frontend/src/components/RecommendationCard.tsx
import React from 'react';
import { RecommendationItem } from '../types';
import { motion } from 'framer-motion';
import Tooltip from './Tooltip';
import { FiInfo } from 'react-icons/fi';

interface RecommendationCardProps {
  recommendation: RecommendationItem;
  rank: number;
}

const cardHoverVariant = {
  hover: {
    scale: 1.03,
    boxShadow: "0px 10px 20px rgba(6, 182, 212, 0.1)", // Primary color shadow
    transition: { duration: 0.2, ease: 'easeOut' } // Smoother ease
  }
};

const RecommendationCard: React.FC<RecommendationCardProps> = ({ recommendation, rank }) => {
  return (
    <motion.div
        className="bg-surface rounded-lg shadow-lg overflow-hidden border border-border-color h-full flex flex-col justify-between"
        layout // Keep layout for smooth list animations
        variants={cardHoverVariant}
        whileHover="hover"
    >
      <div className="p-5">
         {/* Header with Rank and Score */}
         <div className="flex justify-between items-center mb-4 pb-2 border-b border-border-color">
            <span className="text-xs font-bold uppercase tracking-wider text-background bg-primary px-2.5 py-1 rounded-full">
                Rank #{rank}
            </span>
            {/* Score with Tooltip */}
            <Tooltip content="Model's predicted relevance score (higher is better)" position="top">
                 <span className="text-sm font-semibold text-text-secondary flex items-center gap-1 cursor-help">
                    Score: <span className="text-primary font-bold">{recommendation.score.toFixed(3)}</span>
                    <FiInfo size={12} className="opacity-60" />
                 </span>
             </Tooltip>
        </div>

        {/* Main Content */}
        <h3 className="text-lg font-semibold text-text-primary mb-2 truncate" title={recommendation.presentation_id}>
            {recommendation.presentation_id}
        </h3>
        <div className="text-sm text-text-muted space-y-1">
            <p><span className="font-medium text-text-secondary">Module:</span> {recommendation.module_id}</p>
            <p><span className="font-medium text-text-secondary">Presentation:</span> {recommendation.presentation_code}</p>
            {recommendation.module_presentation_length != null && (
                 <p><span className="font-medium text-text-secondary">Duration:</span> {recommendation.module_presentation_length} days</p>
            )}
        </div>
      </div>
    </motion.div>
  );
};

export default RecommendationCard;