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
    const stepVariants = {
        hidden: { opacity: 0, x: -30 },
        visible: {
            opacity: 1,
            x: 0,
            transition: { duration: 0.6, ease: "easeOut", delay: index * 0.1 } // Stagger based on index
        }
    };

    const iconVariants = {
         hidden: { scale: 0.5, opacity: 0 },
         visible: {
             scale: 1,
             opacity: 1,
             transition: { duration: 0.4, ease: "backOut", delay: index * 0.1 + 0.2 } // Slightly delayed pop
         }
    };

    const lineVariants = {
         hidden: { height: 0 },
         visible: {
             height: 'auto', // Animates to the required height
             transition: { duration: 0.5, ease: "easeOut", delay: index * 0.1 + 0.1 } // Coordinated line growth
         }
    };

    // Add subtle alternating background for readability
    const backgroundClass = index % 2 === 0 ? 'bg-surface/20' : 'bg-surface/50';

    return (
        <motion.div
            className="flex relative pb-12 md:pb-16"
            variants={stepVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.3 }} // Trigger animation sooner
        >
            {/* Animated Connecting Line */}
            {!isLast && (
                <motion.div
                    className="absolute left-6 top-12 w-0.5 bg-border-color opacity-50"
                    style={{ bottom: '-2rem' }} // Adjust based on padding-bottom
                    variants={lineVariants}
                    initial="hidden"
                    whileInView="visible"
                    viewport={{ once: true, amount: 0.1 }} // Line starts growing earlier
                />
            )}

            {/* Animated Icon */}
            <motion.div
                className="flex-shrink-0 w-12 h-12 rounded-full bg-primary/10 border-2 border-primary inline-flex items-center justify-center text-primary relative z-10 shadow-md"
                variants={iconVariants} // Use separate variants for icon animation
            >
                {icon}
            </motion.div>

            {/* Content Area */}
            <div className={`flex-grow pl-6 md:pl-10 pr-4 py-4 rounded-r-lg ${backgroundClass}`}>
                <h3 className="font-semibold title-font text-xl md:text-2xl text-text-primary mb-2 tracking-wide">{title}</h3>
                {/* Apply prose styles for better readability if needed */}
                <div className="leading-relaxed text-text-muted text-sm md:text-base prose prose-invert prose-sm md:prose-base max-w-none prose-p:text-text-muted prose-ul:text-text-muted prose-li:text-text-muted">
                     {children}
                 </div>
            </div>
        </motion.div>
    );
};

export default ProcessStep;