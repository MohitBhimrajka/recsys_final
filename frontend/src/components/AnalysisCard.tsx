// frontend/src/components/AnalysisCard.tsx
import React from 'react';
import { motion } from 'framer-motion';
import Tooltip from './Tooltip';
import { FiInfo } from 'react-icons/fi';

interface AnalysisCardProps {
    title: string;
    icon: React.ReactNode;
    tooltipContent?: string; // Optional tooltip for the title info icon
    description?: string; // Optional descriptive text below the title
    children: React.ReactNode; // Content of the card (chart, table, etc.)
    isLoading?: boolean; // Optional loading state
    isEmpty?: boolean; // Optional state to show empty message
    emptyMessage?: string; // Message to show when empty
    className?: string; // Allow custom styling
}

const cardVariant = {
    hidden: { opacity: 0, y: 15 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: "easeOut" } }
};

const AnalysisCard: React.FC<AnalysisCardProps> = ({
    title,
    icon,
    tooltipContent,
    description,
    children,
    isLoading = false,
    isEmpty = false,
    emptyMessage = "No data available for this analysis.",
    className = ''
}) => {
    return (
        <motion.div
            className={`bg-surface/60 rounded-lg border border-border-color/50 p-4 md:p-6 shadow-lg h-full flex flex-col ${className}`} // Use shadow-lg, add flex-col
            variants={cardVariant}
            // Let parent handle initial/animate if part of a staggered group
        >
            {/* Header Section */}
            <div className="flex items-start justify-between mb-3 md:mb-4">
                <div className="flex items-center gap-2">
                    <span className="text-primary flex-shrink-0">{icon}</span>
                    <h3 className="text-base md:text-lg font-semibold text-text-primary">{title}</h3>
                    {tooltipContent && (
                         <Tooltip content={tooltipContent} position="top">
                            <button className='text-text-muted hover:text-primary'><FiInfo size={15} className='mt-px'/></button>
                        </Tooltip>
                    )}
                </div>
            </div>

             {/* Optional Description */}
             {description && (
                <p className="text-xs text-text-muted mb-4">{description}</p>
             )}

            {/* Content Area */}
            <div className="flex-grow"> {/* Allow content to fill height */}
                {isLoading ? (
                     <div className="flex items-center justify-center h-32">
                        {/* Add a simple loading spinner or skeleton */}
                        <div className="animate-spin rounded-full h-6 w-6 border-t-2 border-b-2 border-primary"></div>
                    </div>
                ) : isEmpty ? (
                    <div className="flex items-center justify-center h-32 text-center">
                        <p className="text-sm text-text-muted italic">{emptyMessage}</p>
                    </div>
                ) : (
                    children // Render the actual content (chart, table, etc.)
                )}
            </div>
        </motion.div>
    );
};

export default AnalysisCard;