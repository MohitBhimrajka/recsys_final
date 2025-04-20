// src/components/ModelInfoModal.tsx
import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiX, FiCheckCircle, FiXCircle } from 'react-icons/fi';
import { ModelInfo } from '../data/modelInfo'; // Import the type and data

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
  hidden: { opacity: 0, scale: 0.9 },
  visible: { opacity: 1, scale: 1, transition: { duration: 0.3, ease: 'easeOut' } },
  exit: { opacity: 0, scale: 0.95, transition: { duration: 0.2, ease: 'easeIn' } },
};

const ModelInfoModal: React.FC<ModelInfoModalProps> = ({ isOpen, onClose, model }) => {
  if (!model) return null;

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm"
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
          >
            {/* Header */}
            <div className="flex justify-between items-center p-5 border-b border-border-color">
              <h2 className="text-xl md:text-2xl font-semibold text-primary">{model.name}</h2>
              <button
                onClick={onClose}
                className="text-text-muted hover:text-text-primary transition-colors rounded-full p-1 focus:outline-none focus:ring-2 focus:ring-primary"
                aria-label="Close modal"
              >
                <FiX size={24} />
              </button>
            </div>

            {/* Content */}
            <div className="p-6 max-h-[70vh] overflow-y-auto">
              <p className="text-text-secondary mb-6">{model.description}</p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Pros */}
                <div>
                  <h4 className="text-lg font-semibold text-green-400 mb-3 flex items-center">
                    <FiCheckCircle className="mr-2" /> Strengths
                  </h4>
                  <ul className="list-none space-y-2 text-sm text-text-muted">
                    {model.pros.map((pro, index) => (
                      <li key={`pro-${index}`} className="flex items-start">
                         <span className="text-green-500 mr-2 mt-1 flex-shrink-0">•</span>
                         <span>{pro}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Cons */}
                <div>
                  <h4 className="text-lg font-semibold text-red-400 mb-3 flex items-center">
                    <FiXCircle className="mr-2" /> Weaknesses
                  </h4>
                  <ul className="list-none space-y-2 text-sm text-text-muted">
                    {model.cons.map((con, index) => (
                       <li key={`con-${index}`} className="flex items-start">
                         <span className="text-red-500 mr-2 mt-1 flex-shrink-0">•</span>
                         <span>{con}</span>
                      </li>
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