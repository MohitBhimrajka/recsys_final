// frontend/src/components/ModelInfoModal.tsx
// No changes needed in this phase, file provided for completeness.
import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiX, FiCheckCircle, FiXCircle } from 'react-icons/fi';
import { ModelInfo } from '../types'; // Make sure ModelInfo type path is correct

interface ModelInfoModalProps {
    isOpen: boolean;
    onClose: () => void;
    model: ModelInfo | null;
}

const backdropVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { duration: 0.3 } },
};

const modalVariants = {
    hidden: { opacity: 0, scale: 0.95, y: 20 },
    visible: { opacity: 1, scale: 1, y: 0, transition: { duration: 0.3, ease: 'easeOut' } },
    exit: { opacity: 0, scale: 0.95, y: 10, transition: { duration: 0.2, ease: 'easeIn' } },
};

const ModelInfoModal: React.FC<ModelInfoModalProps> = ({ isOpen, onClose, model }) => {
    if (!model) return null;

    return (
        <AnimatePresence>
            {isOpen && (
                <motion.div
                    className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm" // Slightly darker backdrop
                    variants={backdropVariants}
                    initial="hidden"
                    animate="visible"
                    exit="hidden"
                    onClick={onClose} // Close when clicking backdrop
                >
                    <motion.div
                        className="bg-surface w-full max-w-2xl rounded-xl shadow-2xl border border-border-color overflow-hidden"
                        variants={modalVariants}
                        initial="hidden"
                        animate="visible"
                        exit="exit"
                        onClick={(e) => e.stopPropagation()} // Prevent closing when clicking modal content
                        role="dialog"
                        aria-modal="true"
                        aria-labelledby="model-modal-title"
                    >
                        {/* Header */}
                        <div className="flex justify-between items-center p-5 border-b border-border-color bg-background/30"> {/* Subtle header bg */}
                            <h2 id="model-modal-title" className="text-xl md:text-2xl font-semibold text-primary">{model.name}</h2>
                            <button
                                onClick={onClose}
                                className="text-text-muted hover:text-text-primary transition-colors rounded-full p-1 outline-none focus-visible:ring-2 focus-visible:ring-primary" // Added focus style
                                aria-label="Close modal"
                            >
                                <FiX size={24} />
                            </button>
                        </div>

                        {/* Content */}
                        <div className="p-6 max-h-[70vh] overflow-y-auto">
                            {/* Description with more spacing */}
                            <p className="text-text-secondary mb-8 text-base leading-relaxed">{model.description}</p>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                {/* Pros Section with subtle background */}
                                <div className="bg-green-900/10 p-4 rounded-lg border border-green-700/20">
                                    <h4 className="text-lg font-semibold text-green-400 mb-3 flex items-center">
                                        <FiCheckCircle className="mr-2 flex-shrink-0" /> Strengths {/* Added flex-shrink-0 */}
                                    </h4>
                                    <ul className="list-none space-y-2 text-sm text-text-muted">
                                        {model.pros.map((pro, index) => (
                                            <motion.li
                                                key={`pro-${index}`}
                                                className="flex items-start"
                                                initial={{ opacity: 0, x: -10 }}
                                                animate={{ opacity: 1, x: 0 }}
                                                transition={{ delay: index * 0.05 + 0.1 }} // Stagger list items
                                            >
                                                <span className="text-green-500 mr-2 mt-1 flex-shrink-0">•</span>
                                                <span>{pro}</span>
                                            </motion.li>
                                        ))}
                                    </ul>
                                </div>

                                {/* Cons Section with subtle background */}
                                <div className="bg-red-900/10 p-4 rounded-lg border border-red-700/20">
                                    <h4 className="text-lg font-semibold text-red-400 mb-3 flex items-center">
                                        <FiXCircle className="mr-2 flex-shrink-0" /> Weaknesses {/* Added flex-shrink-0 */}
                                    </h4>
                                    <ul className="list-none space-y-2 text-sm text-text-muted">
                                        {model.cons.map((con, index) => (
                                            <motion.li
                                                key={`con-${index}`}
                                                className="flex items-start"
                                                initial={{ opacity: 0, x: -10 }}
                                                animate={{ opacity: 1, x: 0 }}
                                                transition={{ delay: index * 0.05 + 0.15 }} // Stagger list items
                                            >
                                                <span className="text-red-500 mr-2 mt-1 flex-shrink-0">•</span>
                                                <span>{con}</span>
                                            </motion.li>
                                        ))}
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>
    );
};

export default ModelInfoModal;