// frontend/src/pages/AboutPage.tsx
import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FiDatabase, FiFilter, FiTrendingUp, FiCheckCircle, FiXCircle, FiInfo, FiCode, FiGithub, FiExternalLink, FiLayers, FiCpu, FiUsers, FiBox, FiAlertTriangle, FiArrowRight } from 'react-icons/fi'; // Added FiArrowRight
import ProcessStep from '../components/ProcessStep';
import ModelInfoModal from '../components/ModelInfoModal'; // Import modal
import { modelInfos, ModelInfo, findModelInfoByName } from '../types'; // Import model info

// Reusable list item animation variant
const listItemVariant = {
    hidden: { opacity: 0, y: 10 },
    visible: (i: number) => ({ // Custom prop 'i' for stagger index
        opacity: 1,
        y: 0,
        transition: { delay: i * 0.08, duration: 0.4, ease: "easeOut" }, // Slightly faster stagger
    }),
};

// Component to make model names clickable
const ClickableModelName: React.FC<{ name: string; onClick: (name: string) => void }> = ({ name, onClick }) => (
    <button
        onClick={() => onClick(name)}
        className="text-primary hover:text-primary-light underline decoration-dotted underline-offset-2 font-semibold transition-colors duration-150 inline-flex items-center gap-1 focus:outline-none focus-visible:ring-1 focus-visible:ring-primary rounded"
        title={`Learn more about ${name}`}
    >
       {name} <FiInfo size={13} className="opacity-70"/>
    </button>
);

const AboutPage: React.FC = () => {
    // Modal State
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [currentModalModel, setCurrentModalModel] = useState<ModelInfo | null>(null);

    const openModelInfoModal = (modelName: string) => {
        const modelInfo = findModelInfoByName(modelName);
        if (modelInfo) {
            setCurrentModalModel(modelInfo);
            setIsModalOpen(true);
        }
    };
    const closeModelInfoModal = () => setIsModalOpen(false);

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
                 <ProcessStep index={0} icon={<FiDatabase size={24} />} title="1. Data Foundation (OULAD)">
                    We begin with the
                    <a href="https://analyse.kmi.open.ac.uk/open_dataset" target="_blank" rel="noopener noreferrer" className='mx-1'>Open University Learning Analytics Dataset</a>.
                    The goal is to recommend relevant <strong className='text-text-secondary'>course presentations</strong>. A 'presentation' represents a specific offering of a course module during a particular semester (e.g., module <code>'AAA'</code> offered in semester <code>'2013J'</code>, identified as <code className='text-xs'>AAA_2013J</code>).
                    <br/><br/>
                    Key data sources include:
                    <ul className="list-disc list-inside text-sm mt-2 space-y-1 pl-2">
                        <li><strong className='text-text-secondary'>Student VLE Interactions (<code>studentVle.csv</code>):</strong> Records of student clicks on course materials, forums, quizzes, etc. This is the <strong className='text-text-secondary'>primary signal</strong> for engagement.</li>
                        <li>Student Demographics (<code>studentInfo.csv</code>)</li>
                        <li>Course Structures & Presentations (<code>courses.csv</code>)</li>
                        <li>Student Registrations (<code>studentRegistration.csv</code>)</li>
                    </ul>
                </ProcessStep>

                 <ProcessStep index={1} icon={<FiFilter size={24} />} title="2. Preprocessing & Filtering">
                    Raw data is cleaned (handling missing values, correcting types). VLE interactions are filtered to match active registration periods.
                    <br/><br/>
                    <strong className='text-text-secondary'>Crucially, to improve model stability and focus on engaged users/items:</strong>
                     <ul className="list-disc list-inside text-sm mt-2 space-y-1 pl-2">
                         <li>Users with fewer than 5 interactions were removed.</li>
                         <li>Course presentations interacted with by fewer than 5 users were removed.</li>
                    </ul>
                     <div className="mt-3 p-3 bg-yellow-900/10 border border-yellow-700/30 rounded-md text-sm">
                        <FiAlertTriangle className="inline mr-1 text-yellow-400 mb-1"/>
                        <strong className='text-yellow-200'>Resulting Dataset Context:</strong> This filtering significantly reduced the data scope. The models in this demo were ultimately trained on interactions involving only <strong className='underline decoration-dotted'>22 unique course presentations</strong>. Understanding this is key to interpreting the recommendations' diversity (or lack thereof).
                     </div>
                 </ProcessStep>

                 <ProcessStep index={2} icon={<FiTrendingUp size={24} />} title="3. Quantifying Engagement">
                    Filtered VLE clicks are summed for each student-presentation pair. We calculate an <code>implicit_feedback</code> score using <code>log1p(total_clicks)</code>. This log transformation dampens the effect of extremely high click counts, giving more weight to initial engagement and creating a more balanced signal for the models. Higher scores indicate stronger perceived engagement.
                </ProcessStep>

                 <ProcessStep index={3} icon={<FiCpu size={24} />} title="4. Learning Interaction Patterns">
                    Multiple algorithms analyze the processed <code>implicit_feedback</code> data:
                    <ul className="list-disc list-inside text-sm mt-3 space-y-1.5 pl-2">
                         <motion.li custom={0} variants={listItemVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.5 }}>
                            <ClickableModelName name="ItemCF" onClick={openModelInfoModal} />: Calculates course similarity based on co-interaction patterns.
                         </motion.li>
                         <motion.li custom={1} variants={listItemVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.5 }}>
                            <ClickableModelName name="ALS (f=100)" onClick={openModelInfoModal} />: Matrix factorization finding latent user/course features.
                         </motion.li>
                          <motion.li custom={2} variants={listItemVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.5 }}>
                            <ClickableModelName name="NCF (e=15)" onClick={openModelInfoModal} /> / <ClickableModelName name="Hybrid (e=15)" onClick={openModelInfoModal} />: Neural networks modeling complex interactions.
                          </motion.li>
                     </ul>
                </ProcessStep>

                 <ProcessStep index={4} icon={<FiCheckCircle size={24} />} title="5. Generating Scores">
                     When a student ID is selected in the demo:
                     <ul className="list-disc list-inside text-sm mt-3 space-y-1.5 pl-2">
                        <motion.li custom={0} variants={listItemVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.5 }}>Each trained model predicts a relevance score for candidate courses (those the student hasn't seen).</motion.li>
                        <motion.li custom={1} variants={listItemVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.5 }}>An <strong className='text-text-secondary'>ensemble score</strong> is calculated by combining the normalized scores from each model, weighted by their offline performance (giving more weight to <ClickableModelName name="ItemCF" onClick={openModelInfoModal} />).</motion.li>
                     </ul>
                </ProcessStep>

                 <ProcessStep index={5} icon={<FiLayers size={24} />} title="6. Ranking & Recommendation" isLast={true}>
                    Candidate courses are ranked based on the final <strong className='text-text-secondary'>ensemble score</strong>. The top 9 highest-scoring, previously unseen courses are displayed as the primary "Combined Suggestion". The demo also shows the top 9 from each individual model for comparison purposes.
                </ProcessStep>
            </div>

            {/* Code Link Section */}
            <motion.div
                className="mt-24 text-center"
                initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true, amount: 0.3 }} transition={{ duration: 0.6, delay: 0.2 }}
            >
                <h2 className="text-2xl md:text-3xl font-bold text-text-primary mb-4 flex items-center justify-center gap-2">
                    <FiCode className="text-primary" /> Curious About the Code?
                </h2>
                <p className="text-text-secondary max-w-xl mx-auto mb-8">
                    Dive deeper into the implementation details, explore the different models, or see the data processing pipeline in action.
                </p>
                <div className="flex flex-col sm:flex-row justify-center items-center gap-4">
                    <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                        <Link to="/code-explorer" className="btn btn-secondary w-full sm:w-auto">
                            Explore Project Structure <FiExternalLink className="inline ml-1" />
                        </Link>
                    </motion.div>
                    <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                        <a href="https://github.com/mohitbhimrajka/recsys_final" target="_blank" rel="noopener noreferrer" className="btn btn-outline w-full sm:w-auto">
                            View on GitHub <FiGithub className="inline ml-1" />
                        </a>
                    </motion.div>
                </div>
            </motion.div>

            {/* Limitations Section */}
            <motion.div
                className="mt-24 p-8 md:p-10 bg-black/20 rounded-xl shadow-xl border border-border-color/70 max-w-4xl mx-auto"
                initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true, amount: 0.2 }} transition={{ duration: 0.6 }}
            >
                <h2 className="text-2xl md:text-3xl font-bold text-center mb-8 text-text-primary flex items-center justify-center gap-2">
                    <FiAlertTriangle className="text-yellow-400" /> Important Considerations
                </h2>
                <motion.ul className="list-none space-y-4 text-text-secondary text-sm md:text-base">
                    {/* Added presentation clarification */}
                    <motion.li custom={2} variants={listItemVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.5 }} className="flex items-start">
                        <FiBox className="text-blue-400 mr-3 mt-1 flex-shrink-0" size={18} />
                        <span><strong className="text-text-primary">Small Item Pool:</strong> Due to data filtering for model stability, only <strong className='underline decoration-dotted'>22 unique course presentations</strong> remained. This significantly limits the diversity of recommendations.</span>
                    </motion.li>
                    {/* Other limitations kept */}
                    <motion.li custom={0} variants={listItemVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.5 }} className="flex items-start">
                        <FiUsers className="text-red-500 mr-3 mt-1 flex-shrink-0" size={18} />
                        <span><strong className="text-text-primary">Cold Start:</strong> New students won't receive personalized recommendations initially.</span>
                    </motion.li>
                    <motion.li custom={1} variants={listItemVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.5 }} className="flex items-start">
                        <FiTrendingUp className="text-yellow-500 mr-3 mt-1 flex-shrink-0" size={18} />
                        <span><strong className="text-text-primary">Implicit Feedback Nuance:</strong> Click counts (even log-transformed) assume clicks equal positive interest.</span>
                    </motion.li>
                    <motion.li custom={3} variants={listItemVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.5 }} className="flex items-start">
                        <FiDatabase className="text-green-500 mr-3 mt-1 flex-shrink-0" size={18} />
                        <span><strong className="text-text-primary">Static Data:</strong> The demo models use a fixed OULAD snapshot; real systems need retraining.</span>
                    </motion.li>
                    <motion.li custom={4} variants={listItemVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.5 }} className="flex items-start">
                         <FiLayers className="text-purple-400 mr-3 mt-1 flex-shrink-0" size={18} />
                         <span><strong className="text-text-primary">Simple Ensemble:</strong> Uses a basic weighted average; more advanced techniques could be explored.</span>
                    </motion.li>
                </motion.ul>
            </motion.div>

             {/* Modal for Model Info */}
             <ModelInfoModal
                 isOpen={isModalOpen}
                 onClose={closeModelInfoModal}
                 model={currentModalModel}
             />
        </div>
    );
};

export default AboutPage;