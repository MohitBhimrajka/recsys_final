// frontend/src/components/PresentationDetailModal.tsx
import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiX } from 'react-icons/fi';
import { PresentationDetailInfo } from '../types'; // Use the type alias or RecommendationItem

interface PresentationDetailModalProps {
    isOpen: boolean;
    onClose: () => void;
    presentation: PresentationDetailInfo | null;
}

const backdropVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { duration: 0.3 } },
};

const modalVariants = {
    hidden: { opacity: 0, scale: 0.9, y: 10 },
    visible: { opacity: 1, scale: 1, y: 0, transition: { duration: 0.3, ease: 'easeOut' } },
    exit: { opacity: 0, scale: 0.9, y: 5, transition: { duration: 0.2, ease: 'easeIn' } },
};

const PresentationDetailModal: React.FC<PresentationDetailModalProps> = ({ isOpen, onClose, presentation }) => {
    if (!presentation) return null;

    return (
        <AnimatePresence>
            {isOpen && (
                <motion.div
                    className="fixed inset-0 z-[101] flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm" // Slightly less opaque backdrop than model info
                    variants={backdropVariants}
                    initial="hidden"
                    animate="visible"
                    exit="hidden"
                    onClick={onClose}
                >
                    <motion.div
                        className="bg-surface w-full max-w-md rounded-xl shadow-xl border border-border-color overflow-hidden" // Smaller max-width
                        variants={modalVariants}
                        initial="hidden"
                        animate="visible"
                        exit="exit"
                        onClick={(e) => e.stopPropagation()}
                        role="dialog"
                        aria-modal="true"
                        aria-labelledby="presentation-detail-title"
                    >
                        {/* Header */}
                        <div className="flex justify-between items-center p-4 border-b border-border-color/70">
                            <h2 id="presentation-detail-title" className="text-lg font-semibold text-primary truncate pr-2" title={presentation.presentation_id}>
                                Details: {presentation.presentation_id}
                            </h2>
                            <button
                                onClick={onClose}
                                className="text-text-muted hover:text-text-primary transition-colors rounded-full p-1 outline-none focus-visible:ring-1 focus-visible:ring-primary"
                                aria-label="Close details"
                            >
                                <FiX size={20} />
                            </button>
                        </div>

                        {/* Content */}
                        <div className="p-5 space-y-3 text-sm">
                            <div className="flex justify-between items-center">
                                <span className="text-text-muted">Module ID:</span>
                                <span className="text-text-secondary font-medium">{presentation.module_id}</span>
                            </div>
                             <div className="flex justify-between items-center">
                                <span className="text-text-muted">Presentation Code:</span>
                                <span className="text-text-secondary font-medium">{presentation.presentation_code}</span>
                            </div>
                             {presentation.module_presentation_length != null && (
                                 <div className="flex justify-between items-center">
                                     <span className="text-text-muted">Duration (Days):</span>
                                     <span className="text-text-secondary font-medium">{presentation.module_presentation_length}</span>
                                 </div>
                             )}
                             <div className="flex justify-between items-center border-t border-border-color/50 pt-3 mt-3">
                                <span className="text-text-secondary font-semibold">Predicted Score:</span>
                                <span className="text-primary font-bold text-base">
                                     {presentation.score.toFixed(4)}
                                </span>
                            </div>
                        </div>
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>
    );
};

export default PresentationDetailModal;