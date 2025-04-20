// frontend/src/components/SkeletonCard.tsx
import React from 'react';
import Skeleton from 'react-loading-skeleton';
// Make sure you have imported the CSS for react-loading-skeleton
// usually in main.tsx or App.tsx: import 'react-loading-skeleton/dist/skeleton.css'

const SkeletonCard: React.FC = () => {
  return (
    // Match the general structure and padding of RecommendationCard
    <div className="bg-white rounded-lg shadow-md overflow-hidden border border-gray-200">
        <div className="p-5">
            {/* Header */}
            <div className="flex justify-between items-center mb-3 pb-2 border-b border-gray-200">
                <Skeleton width={80} height={20} inline={true} className="rounded-full" /> {/* Rank */}
                <Skeleton width={100} height={16} /> {/* Score */}
            </div>

            {/* Main Content */}
            <div className="mb-1">
                 <Skeleton height={20} width={`80%`} /> {/* Title */}
            </div>
            <div className="text-sm space-y-1">
                <p><Skeleton width={`60%`} /></p> {/* Detail line 1 */}
                <p><Skeleton width={`50%`} /></p> {/* Detail line 2 */}
                <p><Skeleton width={`70%`} /></p> {/* Detail line 3 */}
            </div>
        </div>
    </div>
  );
};

export default SkeletonCard;