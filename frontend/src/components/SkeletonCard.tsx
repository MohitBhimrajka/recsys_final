// frontend/src/components/SkeletonCard.tsx
import React from 'react';
import Skeleton, { SkeletonTheme } from 'react-loading-skeleton';
// Ensure CSS is imported in App.tsx or main.tsx: import 'react-loading-skeleton/dist/skeleton.css'

const SkeletonCard: React.FC = () => {
  // Define colors based on the black theme
  const baseColor = "#1f2937"; // surface
  const highlightColor = "#374151"; // border-color

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
                <div className="text-sm space-y-2 mt-3"> {/* Increased space */}
                    <p><Skeleton width={`65%`} /></p> {/* Detail line 1 */}
                    <p><Skeleton width={`55%`} /></p> {/* Detail line 2 */}
                    <p><Skeleton width={`75%`} /></p> {/* Detail line 3 */}
                </div>
            </div>
        </div>
    </SkeletonTheme>
  );
};

export default SkeletonCard;