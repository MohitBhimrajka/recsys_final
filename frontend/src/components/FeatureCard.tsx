// frontend/src/components/FeatureCard.tsx
import React from 'react';
import { motion } from 'framer-motion';

interface FeatureCardProps {
    icon: React.ReactNode;
    title: string;
    children: React.ReactNode;
    index?: number; // Optional index for staggered animation delay
}

const FeatureCard: React.FC<FeatureCardProps> = ({ icon, title, children, index = 0 }) => (
    <motion.div
        className="group bg-surface p-6 rounded-xl shadow-xl border border-border-color text-center h-full flex flex-col transition-all duration-300 ease-out hover:border-primary/50 hover:shadow-primary/10 hover:-translate-y-2" // Slightly more lift on hover
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.3 }}
        transition={{ duration: 0.5, ease: 'easeOut', delay: index * 0.1 }} // Add staggered delay based on index
    >
        <div className="flex-grow">
            {/* Icon with hover animation */}
            <motion.div
                className="text-primary text-5xl mb-5 inline-block transition-transform duration-300 ease-out group-hover:scale-110 group-hover:rotate-[-5deg]" // Added slight rotate on hover
            >
                {icon}
            </motion.div>
            <h3 className="text-xl font-semibold text-text-primary mb-3 transition-colors duration-300 group-hover:text-primary">
                {title}
            </h3>
            <p className="text-sm text-text-muted">{children}</p> {/* Ensure text-muted is used for consistency */}
        </div>
    </motion.div>
);

export default FeatureCard;