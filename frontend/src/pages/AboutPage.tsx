// frontend/src/pages/AboutPage.tsx
import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FiDatabase, FiFilter, FiTrendingUp, FiCheckCircle, FiXCircle, FiInfo, FiCode, FiGithub, FiExternalLink, FiLayers, FiCpu } from 'react-icons/fi'; // Added/kept icons
import ProcessStep from '../components/ProcessStep'; // Import the component

// Reusable list item animation variant
const listItemVariant = {
    hidden: { opacity: 0, y: 10 },
    visible: (i: number) => ({ // Custom prop 'i' for stagger index
        opacity: 1,
        y: 0,
        transition: { delay: i * 0.1, duration: 0.4, ease: "easeOut" },
    }),
};

const AboutPage: React.FC = () => {
    return (
        <div className="py-16 md:py-24">
            <motion.h1
                className="text-4xl md:text-5xl font-bold text-center mb-16 md:mb-20 text-text-primary"
                initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
            >
                The Recommendation Process Explained
            </motion.h1>

            {/* Vertical Stepper/Timeline Container */}
            <div className="relative max-w-3xl mx-auto">
                {/* STEP 1 (Unchanged) */}
                <ProcessStep index={0} icon={<FiDatabase size={24} />} title="1. Data Foundation (OULAD)">
                    We start with the Open University Learning Analytics Dataset, focusing on tables detailing student demographics (<code>studentInfo</code>),
                    course structure (<code>courses</code>), registrations (<code>studentRegistration</code>), and detailed VLE interactions (<code>studentVle</code>).
                </ProcessStep>

                {/* STEP 2 (Unchanged) */}
                <ProcessStep index={1} icon={<FiFilter size={24} />} title="2. Preprocessing & Filtering">
                    Raw data is cleaned (handling missing values, correcting types). VLE interactions are filtered to match active registration periods.
                    Critically, sparse data is reduced by removing users and items (course presentations) with very few interactions (typically {'<'} 5), improving model stability.
                </ProcessStep>

                {/* STEP 3 (Unchanged) */}
                <ProcessStep index={2} icon={<FiTrendingUp size={24} />} title="3. Quantifying Engagement">
                    Filtered VLE clicks are aggregated for each student-presentation pair. We calculate an <code className="bg-surface text-primary/80 text-xs px-1.5 py-0.5 rounded mx-0.5 border border-border-color">implicit_feedback</code> score using <code className="bg-surface text-primary/80 text-xs px-1.5 py-0.5 rounded mx-0.5 border border-border-color">log1p(total_clicks)</code>. Higher scores indicate stronger engagement with a course presentation.
                </ProcessStep>

                {/* UPDATED STEP 4 */}
                <ProcessStep index={3} icon={<FiCpu size={24} />} title="4. Learning Interaction Patterns (Multi-Model)">
                    Multiple algorithms analyze the processed interaction data:
                    <ul className="list-disc list-inside text-sm text-text-muted mt-3 space-y-1.5 pl-2">
                         <motion.li custom={0} variants={listItemVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.5 }}>
                            <strong>ItemCF:</strong> Calculates item similarity based on co-occurrence (users interacting with item A also interact with item B).
                         </motion.li>
                         <motion.li custom={1} variants={listItemVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.5 }}>
                             <strong>ALS:</strong> Decomposes the interaction matrix into latent user/item factors.
                         </motion.li>
                          <motion.li custom={2} variants={listItemVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.5 }}>
                              <strong>NCF/Hybrid:</strong> Use neural networks to model complex non-linear interactions, potentially incorporating item features (Hybrid).
                          </motion.li>
                     </ul>
                </ProcessStep>

                 {/* UPDATED STEP 5 */}
                 <ProcessStep index={4} icon={<FiCheckCircle size={24} />} title="5. Generating Individual & Ensemble Scores">
                     When a student ID is selected in the demo:
                     <ul className="list-disc list-inside text-sm text-text-muted mt-3 space-y-1.5 pl-2">
                        <motion.li custom={0} variants={listItemVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.5 }}>Each trained model predicts a relevance score for candidate (unseen) items based on the student's past interactions and the model's learned patterns.</motion.li>
                        <motion.li custom={1} variants={listItemVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.5 }}>The individual model predictions are shown for comparison.</motion.li>
                        <motion.li custom={2} variants={listItemVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.5 }}>An <strong className='text-text-secondary'>ensemble score</strong> is also calculated by taking a weighted average of the normalized scores from each model (weights based on evaluation performance).</motion.li>
                     </ul>
                </ProcessStep>

                 {/* UPDATED STEP 6 */}
                 <ProcessStep index={5} icon={<FiLayers size={24} />} title="6. Ranking & Recommendation" isLast={true}>
                    Candidate items are ranked based on the final <strong className='text-text-secondary'>ensemble score</strong>. The top-K highest-scoring, previously unseen items are displayed as the primary personalized recommendations. The demo also shows the top-K from each individual model.
                </ProcessStep>
            </div>

            {/* Code Link Section (Unchanged) */}
            <motion.div
                className="mt-24 text-center"
                initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true, amount: 0.3 }} transition={{ duration: 0.6, delay: 0.2 }}
            >
                <h2 className="text-2xl md:text-3xl font-bold text-text-primary mb-4">
                    <FiCode className="inline mr-2 text-primary" /> Curious About the Code?
                </h2>
                <p className="text-text-secondary max-w-xl mx-auto mb-8">
                    Dive deeper into the implementation details, explore the different models, or see the data processing pipeline.
                </p>
                <div className="flex flex-col sm:flex-row justify-center items-center gap-4">
                    <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                        <Link to="/code-explorer" className="btn btn-secondary w-full sm:w-auto focus:outline-none focus-visible:ring-4 focus-visible:ring-border-color/50">
                            Explore Project Structure <FiExternalLink className="inline ml-2" />
                        </Link>
                    </motion.div>
                    <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                        <a href="https://github.com/mohitbhimrajka/recsys_final" target="_blank" rel="noopener noreferrer" className="btn btn-outline w-full sm:w-auto focus:outline-none focus-visible:ring-4 focus-visible:ring-primary/50">
                            View on GitHub <FiGithub className="inline ml-2" />
                        </a>
                    </motion.div>
                </div>
            </motion.div>

            {/* Limitations Section (Added Ensemble Limitation) */}
            <motion.div
                className="mt-24 p-8 md:p-10 bg-black/20 rounded-xl shadow-xl border border-border-color/70 max-w-4xl mx-auto"
                initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true, amount: 0.2 }} transition={{ duration: 0.6 }}
            >
                <h2 className="text-2xl md:text-3xl font-bold text-center mb-8 text-text-primary">
                    <FiInfo className="inline mr-2 text-primary" /> Important Considerations
                </h2>
                <motion.ul className="list-none space-y-4 text-text-secondary text-sm md:text-base">
                    <motion.li custom={0} variants={listItemVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.5 }} className="flex items-start">
                        <FiXCircle className="text-red-500 mr-3 mt-1 flex-shrink-0" size={18} />
                        <span><strong className="text-text-primary">Cold Start:</strong> New users or new courses cannot be handled well by most models without retraining or specific strategies (like content-based or popularity fallback).</span>
                    </motion.li>
                    <motion.li custom={1} variants={listItemVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.5 }} className="flex items-start">
                        <FiXCircle className="text-yellow-500 mr-3 mt-1 flex-shrink-0" size={18} />
                        <span><strong className="text-text-primary">Implicit Feedback:</strong> Click counts (<code>log1p</code>) are a proxy for interest, not a perfect measure of satisfaction or learning. Explicit feedback (ratings) would be stronger if available.</span>
                    </motion.li>
                    <motion.li custom={2} variants={listItemVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.5 }} className="flex items-start">
                        <FiXCircle className="text-blue-400 mr-3 mt-1 flex-shrink-0" size={18} />
                        <span><strong className="text-text-primary">Data Snapshot:</strong> The demo models are trained on a specific OULAD snapshot. A real-world system needs periodic retraining.</span>
                    </motion.li>
                    <motion.li custom={3} variants={listItemVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.5 }} className="flex items-start">
                        <FiXCircle className="text-green-500 mr-3 mt-1 flex-shrink-0" size={18} />
                        <span><strong className="text-text-primary">Item Pool Size:</strong> Preprocessing resulted in 22 unique course presentations, limiting recommendation diversity.</span>
                    </motion.li>
                    {/* ADDED ENSEMBLE LIMITATION */}
                     <motion.li custom={4} variants={listItemVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.5 }} className="flex items-start">
                         <FiXCircle className="text-purple-400 mr-3 mt-1 flex-shrink-0" size={18} />
                         <span><strong className="text-text-primary">Ensemble Simplicity:</strong> The current ensemble uses a basic weighted average based on prior evaluation. More sophisticated methods (e.g., learning weights, rank aggregation) exist but add complexity.</span>
                    </motion.li>
                </motion.ul>
            </motion.div>
        </div>
    );
};

export default AboutPage;