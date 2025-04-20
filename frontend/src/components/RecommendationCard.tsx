// frontend/src/components/RecommendationCard.tsx
import React from 'react';
import { RecommendationItem } from '../types';

interface RecommendationCardProps {
  recommendation: RecommendationItem;
  rank: number;
}

const RecommendationCard: React.FC<RecommendationCardProps> = ({ recommendation, rank }) => {
  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden transition-transform duration-200 hover:scale-105">
      <div className="p-4">
         <div className="flex justify-between items-baseline mb-2">
            <h3 className="text-lg font-semibold text-indigo-700">
                #{rank}: {recommendation.presentation_id}
            </h3>
            <span className="text-sm font-medium text-gray-600 bg-indigo-100 px-2 py-0.5 rounded">
                Score: {recommendation.score.toFixed(4)}
             </span>
        </div>
        <p className="text-sm text-gray-600">Module: {recommendation.module_id}</p>
        <p className="text-sm text-gray-600">Code: {recommendation.presentation_code}</p>
        {recommendation.module_presentation_length && ( // Only display if length exists
             <p className="text-sm text-gray-500 mt-1">Length: {recommendation.module_presentation_length} days</p>
        )}
      </div>
    </div>
  );
};

export default RecommendationCard;