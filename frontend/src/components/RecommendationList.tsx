// frontend/src/components/RecommendationList.tsx
import React from 'react';
import { RecommendationItem } from '../types';
import RecommendationCard from './RecommendationCard';

interface RecommendationListProps {
  recommendations: RecommendationItem[];
  selectedUserId: number | null;
}

const RecommendationList: React.FC<RecommendationListProps> = ({ recommendations, selectedUserId }) => {
  if (!selectedUserId) {
      return <p className="text-center text-gray-500 mt-4">Please enter a student ID to see recommendations.</p>;
  }

  if (recommendations.length === 0) {
    return <p className="text-center text-gray-500 mt-4">No recommendations found for student ID {selectedUserId}. They might be new or have seen all applicable courses.</p>;
  }

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4 text-center">Top {recommendations.length} Recommendations for Student {selectedUserId}</h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {recommendations.map((rec, index) => (
          <RecommendationCard key={rec.presentation_id} recommendation={rec} rank={index + 1} />
        ))}
      </div>
    </div>
  );
};

export default RecommendationList;