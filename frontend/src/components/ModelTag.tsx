// frontend/src/components/ModelTag.tsx
import React from 'react';
import { motion } from 'framer-motion';
import { FiCheckCircle, FiInfo } from 'react-icons/fi';
import { ModelInfo } from '../types'; // Assuming ModelInfo type is defined here or imported

interface ModelTagProps {
    model: ModelInfo;
    onClick: () => void;
    isDemoModel?: boolean;
}

const ModelTag: React.FC<ModelTagProps> = ({ model, onClick, isDemoModel = false }) => {
    const baseClasses = "border px-4 py-1.5 rounded-full text-sm cursor-pointer transition-all duration-200 flex items-center gap-1.5 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-background"; // Added focus styles
    // Enhanced active state
    const activeClasses = "bg-primary/15 border-primary text-primary font-medium shadow-sm relative overflow-hidden";
    const inactiveClasses = "bg-surface border-border-color text-text-muted hover:border-primary/50 hover:text-text-secondary";

    return (
        <motion.button
            onClick={onClick}
            className={`${baseClasses} ${isDemoModel ? activeClasses : inactiveClasses}`}
            whileHover={{ y: -2, scale: 1.03 }} // Slightly adjusted scale
            whileTap={{ scale: 0.97 }} // Adjusted tap scale
            // Add subtle pulse animation to the demo tag
            animate={isDemoModel ? {
                 scale: [1, 1.02, 1],
                 transition: { duration: 1.5, repeat: Infinity, ease: "easeInOut" }
             } : {}}
        >
            {/* Optional: Subtle shimmer effect for demo tag */}
            {isDemoModel && (
                 <motion.div
                     className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-primary/10 to-transparent"
                     animate={{ x: ['-100%', '100%'] }}
                     transition={{ duration: 2, repeat: Infinity, ease: 'linear', delay: 0.5 }}
                 />
             )}
             <span className="relative z-[1]"> {/* Ensure text is above shimmer */}
                 {model.name}
             </span>
             <span className={`relative z-[1] ${isDemoModel ? 'text-primary' : 'opacity-60'}`}>
                 {isDemoModel ? <FiCheckCircle size={14} className="flex-shrink-0"/> : <FiInfo size={14} className="flex-shrink-0"/>}
             </span>
        </motion.button>
    );
};

export default ModelTag;