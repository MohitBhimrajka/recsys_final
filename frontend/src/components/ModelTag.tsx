// frontend/src/components/ModelTag.tsx
import React from 'react';
import { motion } from 'framer-motion';
import { FiCheckCircle, FiInfo } from 'react-icons/fi';
import { ModelInfo } from '../types';

interface ModelTagProps {
    model: ModelInfo;
    onClick: () => void;
    // Renamed isDemoModel to isHighlighted for more general use
    isHighlighted?: boolean;
}

const ModelTag: React.FC<ModelTagProps> = ({ model, onClick, isHighlighted = false }) => {
    const baseClasses = "border px-4 py-1.5 rounded-full text-sm cursor-pointer transition-all duration-200 flex items-center gap-1.5 outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-background"; // Added focus styles
    // Enhanced active/highlighted state
    const activeClasses = "bg-primary/15 border-primary text-primary font-medium shadow-sm relative overflow-hidden";
    const inactiveClasses = "bg-surface border-border-color text-text-muted hover:border-primary/50 hover:text-text-secondary";

    return (
        <motion.button
            onClick={onClick}
            className={`${baseClasses} ${isHighlighted ? activeClasses : inactiveClasses}`}
            whileHover={{ y: -2, scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            // Subtle pulse animation only for the highlighted tag
            animate={isHighlighted ? {
                 scale: [1, 1.02, 1],
                 transition: { duration: 1.8, repeat: Infinity, ease: "easeInOut", delay: 0.2 } // Slightly slower pulse
             } : {}}
            title={`Click to learn more about the ${model.name} model`} // Add tooltip hint
        >
            {/* Optional: Subtle shimmer effect for highlighted tag */}
            {isHighlighted && (
                 <motion.div
                     className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-primary/10 to-transparent opacity-50" // Make shimmer less opaque
                     animate={{ x: ['-100%', '100%'] }}
                     transition={{ duration: 2.5, repeat: Infinity, ease: 'linear', delay: 0.5 }}
                     style={{ mixBlendMode: 'overlay' }} // Try blend mode
                 />
             )}
             {/* Text and Icon */}
             <span className="relative z-[1] flex-shrink-0"> {/* Ensure text is above shimmer */}
                 {model.name}
             </span>
             <span className={`relative z-[1] ${isHighlighted ? 'text-primary' : 'opacity-60'}`}>
                 {/* Show Check icon if highlighted, Info icon otherwise */}
                 {isHighlighted ? <FiCheckCircle size={14} className="flex-shrink-0"/> : <FiInfo size={14} className="flex-shrink-0"/>}
             </span>
        </motion.button>
    );
};

export default ModelTag;