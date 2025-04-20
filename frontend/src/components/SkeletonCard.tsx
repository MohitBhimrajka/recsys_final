// frontend/src/components/SkeletonCard.tsx
import React from 'react';
import Skeleton, { SkeletonTheme } from 'react-loading-skeleton';
// Ensure CSS is imported: import 'react-loading-skeleton/dist/skeleton.css'

const SkeletonCard: React.FC = () => {
  const baseColor = "#111111"; // surface (slightly darker than default surface for skeleton)
  const highlightColor = "#2d2d2d"; // border-color

  return (
    <SkeletonTheme baseColor={baseColor} highlightColor={highlightColor}>
        <div className="bg-surface rounded-lg shadow-lg overflow-hidden border border-border-color h-full">
            <div className="p-5">
                {/* Header */}
                <div className="flex justify-between items-center mb-4 pb-2 border-b border-border-color">
                    <Skeleton width={80} height={20} inline={true} className="rounded-full" /> {/* Rank */}
                    <Skeleton width={110} height={16} /> {/* Score */}
                </div>
                {/* Main Content */}
                <div className="mb-2">
                    <Skeleton height={20} width={`85%`} /> {/* Title */}
                </div>
                <div className="text-sm space-y-2 mt-3">
                    <p><Skeleton width={`65%`} /></p>
                    <p><Skeleton width={`55%`} /></p>
                    <p><Skeleton width={`75%`} /></p>
                </div>
            </div>
        </div>
    </SkeletonTheme>
  );
};

export default SkeletonCard;