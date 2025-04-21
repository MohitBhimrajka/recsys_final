// frontend/src/components/ProcessStep.tsx
import React from 'react';
import { motion } from 'framer-motion';

interface ProcessStepProps {
    icon: React.ReactNode;
    title: string;
    children: React.ReactNode;
    isLast?: boolean;
    index: number; // To stagger animations based on order
}

const ProcessStep: React.FC<ProcessStepProps> = ({ icon, title, children, isLast = false, index }) => {
    // Animation variants
    const stepVariants = {
        hidden: { opacity: 0, x: -30 },
        visible: { opacity: 1, x: 0, transition: { duration: 0.5, ease: [0.25, 1, 0.5, 1], delay: index * 0.15 } } // Use a custom ease-out curve
    };
    const iconVariants = {
         hidden: { scale: 0.3, opacity: 0, rotate: -45 }, // Add rotation
         visible: { scale: 1, opacity: 1, rotate: 0, transition: { type: 'spring', stiffness: 250, damping: 12, delay: index * 0.15 + 0.15 } } // Spring animation
    };
    const lineVariants = {
         hidden: { scaleY: 0, originY: 0 }, // Animate scaleY from top
         visible: { scaleY: 1, originY: 0, transition: { duration: 0.6, ease: "easeOut", delay: index * 0.15 + 0.1 } } // Coordinated line growth
    };

    // Alternating background for visual separation
    const backgroundClass = index % 2 === 0 ? 'bg-surface/40' : 'bg-surface/70'; // Adjusted opacity

    return (
        <motion.div
            className="flex relative pb-12 md:pb-16" // Keep padding for line space
            variants={stepVariants}
            initial="hidden"
            whileInView="visible" // Trigger animation when in view
            viewport={{ once: true, amount: 0.25 }} // Trigger slightly earlier
        >
            {/* --- Vertical Connecting Line --- */}
            {/* Positioned behind the icon using z-index */}
            {!isLast && (
                <motion.div
                    className="absolute left-[23px] top-6 bottom-[-3rem] w-[2px] bg-gradient-to-b from-primary/60 to-primary/10 z-0" // Adjusted left, use gradient
                    variants={lineVariants}
                    initial="hidden"
                    whileInView="visible" // Use whileInView for the line as well
                    viewport={{ once: true, amount: 0.1 }}
                    aria-hidden="true"
                />
            )}

            {/* --- Icon Container --- */}
            {/* Ensure icon is above the line with z-index */}
            <motion.div
                className="flex-shrink-0 w-12 h-12 rounded-full bg-surface border-2 border-primary inline-flex items-center justify-center text-primary relative z-10 shadow-lg shadow-primary/20" // Enhanced shadow
                variants={iconVariants}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true, amount: 0.3 }} // Icon animates when it comes into view
            >
                {icon}
            </motion.div>

            {/* --- Content Area --- */}
            <div className={`flex-grow pl-6 md:pl-10 pr-4 py-4 rounded-r-lg ${backgroundClass}`}>
                <h3 className="font-semibold title-font text-xl md:text-2xl text-text-primary mb-2 tracking-tight">{title}</h3>
                 {/* Ensure prose styling for readability, links, and code blocks */}
                <div className="leading-relaxed text-text-muted text-sm md:text-base prose prose-invert prose-sm md:prose-base max-w-none prose-p:text-text-muted prose-a:text-primary prose-a:font-medium hover:prose-a:text-primary-light focus-visible:prose-a:ring-1 focus-visible:prose-a:ring-primary prose-code:bg-background/70 prose-code:text-primary/90 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-xs prose-code:font-mono prose-code:border prose-code:border-border-color prose-ul:text-text-muted prose-li:text-text-muted prose-strong:text-text-secondary">
                     {children}
                 </div>
            </div>
        </motion.div>
    );
};

export default ProcessStep;