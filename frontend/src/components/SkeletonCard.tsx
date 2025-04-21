// frontend/src/components/SkeletonCard.tsx
import React from 'react';
import Skeleton, { SkeletonTheme } from 'react-loading-skeleton';
// Ensure CSS is imported in main app: import 'react-loading-skeleton/dist/skeleton.css'

const SkeletonCard: React.FC = () => {
  // Using theme colors for skeleton consistency
  const baseColor = "var(--color-surface)"; // #111111
  const highlightColor = "var(--color-border-color)"; // #2d2d2d

  return (
    <SkeletonTheme baseColor={baseColor} highlightColor={highlightColor}>
        <div className="bg-surface rounded-lg shadow-lg overflow-hidden border border-border-color h-full flex flex-col">
            <div className="p-5 flex flex-col flex-grow">
                {/* Header Placeholder - Rank Badge */}
                 <div className="mb-6"> {/* Increased margin bottom */}
                      <Skeleton width={60} height={20} inline={true} className="rounded-full" /> {/* Rank */}
                 </div>

                {/* Main Content Placeholder */}
                <div className="mb-3">
                    <Skeleton height={24} width={`80%`} /> {/* Title - slightly taller */}
                </div>
                 <div className="text-sm space-y-2 mb-4 flex-grow"> {/* Add flex-grow */}
                    <p><Skeleton width={`60%`} /></p> {/* Module */}
                    <p><Skeleton width={`50%`} /></p> {/* Code */}
                    <p><Skeleton width={`70%`} /></p> {/* Duration */}
                </div>

                {/* Score Section Placeholder at the bottom */}
                <div className='mt-auto border-t border-border-color/50 pt-3'>
                     <div className="flex justify-between items-center">
                         <Skeleton width={90} height={16} /> {/* "Predicted Score" label */}
                         <Skeleton width={70} height={20} /> {/* Score value */}
                    </div>
                     {/* Score Bar Removed */}
                </div>
            </div>
        </div>
    </SkeletonTheme>
  );
};

export default SkeletonCard;