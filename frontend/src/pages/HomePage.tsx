// frontend/src/pages/HomePage.tsx
import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { motion, useScroll, useTransform } from 'framer-motion';
import { FiArrowRight, FiDatabase, FiCpu, FiLayers, FiGithub, FiChevronDown, FiBarChart2, FiSettings } from 'react-icons/fi'; // Ensure all needed icons are here
import ModelInfoModal from '../components/ModelInfoModal';
import FeatureCard from '../components/FeatureCard';
import ModelTag from '../components/ModelTag';
import { modelInfos, ModelInfo } from '../types'; // Assuming types are defined here

// --- HomePage Component ---
const HomePage: React.FC = () => {
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null);

    // Scroll animation setup (Kept from original)
    const { scrollYProgress } = useScroll();
    // Example: Make grid fade slightly slower on scroll out
    const gridOpacity = useTransform(scrollYProgress, [0, 0.1, 0.2], [0.03, 0.03, 0]); // Adjust opacity range/threshold if needed

    // Modal open/close functions (Kept from original)
    const openModal = (model: ModelInfo) => {
        setSelectedModel(model);
        setIsModalOpen(true);
    };
    const closeModal = () => {
        setIsModalOpen(false);
        // Delay clearing selectedModel to allow exit animation
        setTimeout(() => setSelectedModel(null), 300);
    };

    // --- Framer Motion Variants (Kept from original, reviewed for phase) ---
    const heroVariant = {
        hidden: { opacity: 0, y: -20 },
        visible: { opacity: 1, y: 0, transition: { duration: 0.8, delay: 0.2, ease: "easeOut" } },
    };
    const buttonGroupVariant = {
        hidden: {}, // Parent doesn't need opacity: 0
        visible: { transition: { staggerChildren: 0.15, delayChildren: 0.5 } }
    };
    const buttonVariant = {
        hidden: { opacity: 0, y: 15 }, // Start slightly lower
        visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: 'easeOut' } }
    };
    const sectionVariant = {
        hidden: { opacity: 0, y: 40 }, // Increase initial distance
        visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: "easeOut" } },
    };
    const scrollIndicatorVariant = {
        hidden: { opacity: 0, y: 10 },
        visible: { opacity: 1, y: 0, transition: { duration: 1, delay: 1.5, ease: "easeOut" } },
        bounce: { y: [0, -8, 0], transition: { duration: 1.5, repeat: Infinity, ease: "easeInOut", delay: 2.5 } },
        fade: { opacity: [1, 0.5, 1], transition: { duration: 2.0, repeat: Infinity, ease: "linear", delay: 2.5 } }
    };

    return (
        <div className="text-text-secondary">
            {/* --- Hero Section --- */}
            <section
                className="min-h-screen flex flex-col justify-center items-center text-center px-4 relative overflow-hidden bg-gradient-to-br from-black via-background to-surface"
            >
                {/* Animated Background Grid */}
                <motion.div
                    className="absolute inset-0 opacity-[0.03] bg-[linear-gradient(to_right,#ffffff12_1px,transparent_1px),linear-gradient(to_bottom,#ffffff12_1px,transparent_1px)] bg-[size:35px_35px] [mask-image:radial-gradient(ellipse_50%_50%_at_50%_50%,#000_70%,transparent_100%)]"
                    style={{ opacity: gridOpacity }}
                 />

                <motion.div variants={heroVariant} initial="hidden" animate="visible" className="z-10">
                    {/* Updated Headline */}
                    <h1 className="text-5xl sm:text-6xl lg:text-7xl font-extrabold text-text-primary mb-5 !leading-tight tracking-tight">
                         Personalized Course Recommendations
                    </h1>
                    {/* === MODIFIED SUB-HEADLINE === */}
                    <p className="text-lg md:text-xl text-text-secondary max-w-3xl mx-auto mb-10">
                        Discover relevant Open University course <strong className='text-text-primary'>presentations</strong> (specific offerings like <code className='text-sm'>AAA_2013J</code>) using recommendations generated from the <FiDatabase className='inline mb-1 mx-1 opacity-70' />
                        <a href="https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">OULAD dataset</a>.
                        This system analyzes millions of anonymized student interactions within the Virtual Learning Environment (VLE) to suggest relevant content.
                    </p>
                    {/* Button Group (Kept structure) */}
                    <motion.div
                        className="flex flex-col sm:flex-row items-center justify-center gap-4"
                        variants={buttonGroupVariant}
                        initial="hidden"
                        animate="visible"
                    >
                        <motion.div variants={buttonVariant} whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                            <Link
                                to="/demo"
                                className="btn btn-primary text-lg px-10 py-4 shadow-primary/40 w-full sm:w-auto" // Ensured btn classes are applied
                            >
                                Launch Demo <FiArrowRight className="inline ml-2" />
                            </Link>
                        </motion.div>
                        <motion.div variants={buttonVariant} whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                            <a
                                href="https://github.com/mohitbhimrajka/recsys_final"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="btn btn-secondary text-lg px-10 py-4 w-full sm:w-auto" // Ensured btn classes are applied
                            >
                                <FiGithub className="inline mr-2" /> View Code
                            </a>
                        </motion.div>
                    </motion.div>
                </motion.div>

                {/* Animated Scroll Down Indicator (Kept) */}
                <motion.div
                    className="absolute bottom-10 text-text-muted text-xs z-10 flex flex-col items-center"
                    variants={scrollIndicatorVariant}
                    initial="hidden"
                    animate={["visible", "bounce", "fade"]}
                >
                    <span>Scroll Down</span>
                    <FiChevronDown size={20} />
                </motion.div>
            </section>

            {/* --- Content Sections Container --- */}
            <div className="container mx-auto px-4 pt-24 pb-16 space-y-24 md:space-y-32">
                {/* Section 1: The Challenge & Data (Refined Text) */}
                <motion.section
                    variants={sectionVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.2 }}
                    className="text-center max-w-4xl mx-auto"
                >
                    <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-text-primary mb-6">The OULAD Dataset: Real Interactions</h2>
                    <p className="text-text-secondary md:text-lg">
                        Navigating online courses can be tough. This project analyzes millions of anonymized student interactions from the OULAD dataset to uncover patterns and guide students towards relevant content based on collective behavior.
                    </p>
                </motion.section>

                {/* Section 2: How it Works (Refined Content & Icons) */}
                <motion.section
                    variants={sectionVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.2 }}
                >
                    <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-text-primary mb-12 text-center">Core Approach: Data to Recommendations</h2>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-8 items-stretch">
                        {/* Updated Feature Cards with stagger index */}
                         <FeatureCard icon={<FiDatabase />} title="1. Process Real Data" index={0}>
                            Cleaned and filtered OULAD interactions, focusing on engagement within active registration periods. Calculated implicit feedback via <code className="text-xs">log1p(clicks)</code>.
                        </FeatureCard>
                        <FeatureCard icon={<FiCpu />} title="2. Train Diverse Models" index={1}>
                            Implemented and trained multiple algorithms (ItemCF, ALS, NCF, Hybrid) to learn different types of interaction patterns from the processed data.
                        </FeatureCard>
                         <FeatureCard icon={<FiLayers />} title="3. Get Personalized Suggestions" index={2}>
                             The demo combines model predictions into a weighted ensemble score and allows comparison against individual model results for transparency.
                        </FeatureCard>
                    </div>
                    {/* Link to About Page (Kept) */}
                    <motion.div
                        className="text-center mt-14"
                        initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true, amount: 0.5 }} transition={{ delay: 0.3 }}
                    >
                        <Link to="/about" className="btn btn-outline text-base px-8">
                            See Detailed Process <FiArrowRight className="inline ml-1" />
                        </Link>
                    </motion.div>
                </motion.section>

                {/* Section 3: Models Explored (Refined Text & ModelTag Usage) */}
                <motion.section
                    className="bg-surface p-8 md:p-12 rounded-xl shadow-xl border border-border-color"
                    variants={sectionVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.2 }}
                >
                    <div className="text-center">
                        <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-text-primary mb-6">Models Explored</h2>
                        <p className="text-text-secondary md:text-lg max-w-3xl mx-auto mb-10">
                             Evaluated several techniques, from simple baselines to neural networks. The demo uses all trained models. <strong className='text-text-primary'>ItemCF showed the strongest performance</strong> in offline tests. Click any model tag below to learn more about its approach.
                        </p>
                        <div className="flex flex-wrap justify-center gap-3 md:gap-4">
                            {modelInfos.map((model) => (
                                <ModelTag
                                    key={model.id}
                                    model={model}
                                    onClick={() => openModal(model)} // Trigger modal on click
                                    // Highlight ItemCF as the best performing based on evaluation
                                    isHighlighted={model.id === 'itemcf'}
                                />
                            ))}
                        </div>
                    </div>
                </motion.section>

                {/* Section 4: Call to Action to Demo (Refined Text) */}
                <motion.section
                    className="text-center"
                    variants={sectionVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.2 }}
                >
                    <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-text-primary mb-6">See It In Action</h2>
                    <p className="text-text-secondary md:text-lg max-w-2xl mx-auto mb-8">
                        Ready to explore? Select a student ID in the demo section to view the combined recommendations and compare individual model outputs side-by-side using the interactive tabs.
                    </p>
                    {/* Go to Demo Button (Kept) */}
                    <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                        <Link to="/demo" className="btn btn-primary text-lg px-10 py-4 shadow-primary/40">
                            Go to Demo <FiArrowRight className="inline ml-2" />
                        </Link>
                    </motion.div>
                </motion.section>
            </div> {/* End Content Sections Container */}

            {/* Modal Component (Kept) */}
            <ModelInfoModal
                isOpen={isModalOpen}
                onClose={closeModal}
                model={selectedModel}
            />
        </div>
    );
};

export default HomePage;