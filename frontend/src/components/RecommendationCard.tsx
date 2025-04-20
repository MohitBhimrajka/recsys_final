// frontend/src/components/RecommendationCard.tsx
import React from 'react';
import { RecommendationItem } from '../types'; // Adjust path if needed
import { motion } from 'framer-motion'; // Import motion

interface RecommendationCardProps {
  recommendation: RecommendationItem;
  rank: number;
}

const RecommendationCard: React.FC<RecommendationCardProps> = ({ recommendation, rank }) => {
  return (
    // Use motion.div for animation from RecommendationList
    <div className="bg-white rounded-lg shadow-lg overflow-hidden border border-gray-200 hover:shadow-xl transition-shadow duration-300 ease-in-out">
      <div className="p-5">
         {/* Header with Rank and Score */}
         <div className="flex justify-between items-center mb-3 pb-2 border-b border-gray-200">
            <span className="text-xs font-bold uppercase tracking-wider text-indigo-600 bg-indigo-100 px-2 py-1 rounded-full">
                Rank #{rank}
            </span>
            <span className="text-sm font-semibold text-gray-700">
                Score: {recommendation.score.toFixed(3)} {/* Slightly fewer decimals */}
             </span>
        </div>

        {/* Main Content */}
        <h3 className="text-lg font-bold text-gray-800 mb-1 truncate" title={recommendation.presentation_id}>
            {recommendation.presentation_id}
        </h3>
        <div className="text-sm text-gray-600 space-y-1">
            <p>
                <span className="font-medium">Module:</span> {recommendation.module_id}
            </p>
            <p>
                 <span className="font-medium">Presentation:</span> {recommendation.presentation_code}
            </p>
            {recommendation.module_presentation_length != null && ( // Check explicitly for null/undefined
                 <p>
                     <span className="font-medium">Duration:</span> {recommendation.module_presentation_length} days
                </p>
            )}
        </div>
      </div>
       {/* Optional Footer/Action Area */}
      {/* <div className="bg-gray-50 px-5 py-3">
        <button className="text-sm text-indigo-600 hover:text-indigo-800 font-medium">View Details</button>
      </div> */}
    </div>
  );
};

export default RecommendationCard;