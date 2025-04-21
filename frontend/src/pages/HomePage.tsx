// frontend/src/pages/HomePage.tsx
import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { motion, useScroll, useTransform } from 'framer-motion'; // Keep scroll/transform hooks
import { FiArrowRight, FiDatabase, FiSettings, FiZap, FiGithub, FiChevronDown, FiLayers, FiBarChart2 } from 'react-icons/fi'; // Added FiLayers
import ModelInfoModal from '../components/ModelInfoModal';
import FeatureCard from '../components/FeatureCard'; // Import FeatureCard
import ModelTag from '../components/ModelTag'; // Import ModelTag
import { modelInfos, ModelInfo } from '../types'; // Assuming types are defined here

// --- HomePage Component ---
const HomePage: React.FC = () => {
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null);

    // Scroll animation setup (optional parallax for grid)
    const { scrollYProgress } = useScroll();
    // Example: Make grid fade slightly slower on scroll out
    const gridOpacity = useTransform(scrollYProgress, [0, 0.1, 0.2], [0.03, 0.03, 0]);

    const openModal = (model: ModelInfo) => {
        setSelectedModel(model);
        setIsModalOpen(true);
    };

    const closeModal = () => {
        setIsModalOpen(false);
        // Delay clearing selectedModel to allow exit animation
        setTimeout(() => setSelectedModel(null), 300);
    };

    // --- Framer Motion Variants (Keep Existing) ---
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
                className="min-h-screen flex flex-col justify-center items-center text-center px-4 relative overflow-hidden bg-gradient-to-br from-black via-background to-surface" // Subtle gradient change
            >
                {/* Animated Background Grid */}
                <motion.div
                    className="absolute inset-0 opacity-[0.03] bg-[linear-gradient(to_right,#ffffff12_1px,transparent_1px),linear-gradient(to_bottom,#ffffff12_1px,transparent_1px)] bg-[size:35px_35px] [mask-image:radial-gradient(ellipse_50%_50%_at_50%_50%,#000_70%,transparent_100%)]" // Mask adds fading edge
                    style={{ opacity: gridOpacity }} // Apply scroll-based opacity transform
                 />

                <motion.div variants={heroVariant} initial="hidden" animate="visible" className="z-10">
                    <h1 className="text-5xl sm:text-6xl lg:text-7xl font-extrabold text-text-primary mb-5 !leading-tight tracking-tight">
                        Unlock Learning Paths
                    </h1>
                    {/* UPDATED HERO TEXT */}
                    <p className="text-lg md:text-xl text-text-secondary max-w-3xl mx-auto mb-10">
                        Explore personalized course recommendations generated from real Open University student data using multiple models, including collaborative filtering and neural networks.
                    </p>
                    <motion.div
                        className="flex flex-col sm:flex-row items-center justify-center gap-4"
                        variants={buttonGroupVariant}
                        initial="hidden"
                        animate="visible"
                    >
                        {/* Primary CTA Button */}
                        <motion.div variants={buttonVariant} whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                            <Link
                                to="/demo"
                                className="btn btn-primary text-lg px-10 py-4 shadow-primary/40 w-full sm:w-auto focus:outline-none focus-visible:ring-4 focus-visible:ring-primary/50" // Added focus style
                            >
                                Launch Demo <FiArrowRight className="inline ml-2" />
                            </Link>
                        </motion.div>
                        {/* Secondary CTA Button */}
                        <motion.div variants={buttonVariant} whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                            <a
                                href="https://github.com/mohitbhimrajka/recsys_final"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="btn btn-secondary text-lg px-10 py-4 w-full sm:w-auto focus:outline-none focus-visible:ring-4 focus-visible:ring-border-color/50" // Added focus style
                            >
                                <FiGithub className="inline mr-2" /> View Code
                            </a>
                        </motion.div>
                    </motion.div>
                </motion.div>

                {/* Animated Scroll Down Indicator */}
                <motion.div
                    className="absolute bottom-10 text-text-muted text-xs z-10 flex flex-col items-center"
                    variants={scrollIndicatorVariant}
                    initial="hidden"
                    animate={["visible", "bounce", "fade"]} // Apply multiple animations
                >
                    <span>Scroll Down</span>
                    <FiChevronDown size={20} />
                </motion.div>
            </section>

            {/* --- Content Sections Container --- */}
            <div className="container mx-auto px-4 pt-24 pb-16 space-y-24 md:space-y-32">
                {/* Section 1: The Challenge & Data */}
                <motion.section
                    variants={sectionVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.2 }}
                    className="text-center max-w-4xl mx-auto"
                >
                    <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-text-primary mb-6">The OULAD Dataset: Real Interactions</h2>
                    <p className="text-text-secondary md:text-lg">
                        Navigating the vast landscape of online courses can be challenging. This project taps into the
                        Open University Learning Analytics Dataset (OULAD), containing millions of anonymized VLE interactions,
                        registrations, and demographics, to uncover patterns and guide students towards relevant content.
                    </p>
                </motion.section>

                {/* Section 2: How it Works (Simplified Features) */}
                <motion.section
                    variants={sectionVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.2 }}
                >
                    {/* UPDATED SECTION TITLE */}
                    <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-text-primary mb-12 text-center">Core Approach: From Data to Recommendations</h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 items-stretch">
                        <FeatureCard icon={<FiDatabase />} title="Data Processing">
                            Raw clickstream data is cleaned, filtered, and aggregated to quantify student engagement (implicit feedback) with each course presentation.
                        </FeatureCard>
                        {/* UPDATED CARD 2 */}
                        <FeatureCard icon={<FiSettings />} title="Diverse Models">
                             Multiple algorithms (ItemCF, ALS, NCF, Hybrid) learn different interaction patterns from the data to predict course relevance.
                        </FeatureCard>
                        {/* UPDATED CARD 3 */}
                         <FeatureCard icon={<FiLayers />} title="Ensemble & Comparison">
                             The demo provides a combined recommendation using a weighted average and allows comparison of individual model results.
                        </FeatureCard>
                    </div>
                    <motion.div
                        className="text-center mt-14"
                        initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true, amount: 0.5 }} transition={{ delay: 0.3 }}
                    >
                        <Link to="/about" className="btn btn-outline text-base px-8 focus:outline-none focus-visible:ring-4 focus-visible:ring-primary/50"> {/* Added focus style */}
                            Learn More Details <FiArrowRight className="inline ml-1" />
                        </Link>
                    </motion.div>
                </motion.section>

                {/* Section 3: Models Explored (Interactive) */}
                <motion.section
                    className="bg-surface p-8 md:p-12 rounded-xl shadow-xl border border-border-color"
                    variants={sectionVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.2 }}
                >
                    <div className="text-center">
                        {/* UPDATED SECTION TITLE */}
                        <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-text-primary mb-6">Models Explored</h2>
                        {/* UPDATED TEXT */}
                        <p className="text-text-secondary md:text-lg max-w-3xl mx-auto mb-10">
                             We evaluated several recommendation techniques. The demo utilizes all trained models, allowing comparison and providing a combined ensemble result. Click a model to learn more.
                        </p>
                        <div className="flex flex-wrap justify-center gap-3 md:gap-4">
                            {modelInfos.map((model) => (
                                <ModelTag
                                    key={model.id}
                                    model={model}
                                    onClick={() => openModal(model)}
                                    // Highlight ItemCF as the best performing individual model
                                    isDemoModel={model.id === 'itemcf'}
                                />
                            ))}
                        </div>
                    </div>
                </motion.section>

                {/* Section 4: Call to Action to Demo */}
                <motion.section
                    className="text-center"
                    variants={sectionVariant} initial="hidden" whileInView="visible" viewport={{ once: true, amount: 0.2 }}
                >
                    <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-text-primary mb-6">See It In Action</h2>
                     {/* UPDATED TEXT */}
                    <p className="text-text-secondary md:text-lg max-w-2xl mx-auto mb-8">
                        Ready to explore? Select a student ID in the demo section to view combined recommendations and compare individual model outputs.
                    </p>
                    <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                        <Link to="/demo" className="btn btn-primary text-lg px-10 py-4 shadow-primary/40 focus:outline-none focus-visible:ring-4 focus-visible:ring-primary/50"> {/* Added focus style */}
                            Go to Demo
                        </Link>
                    </motion.div>
                </motion.section>
            </div> {/* End Content Sections Container */}

            <ModelInfoModal
                isOpen={isModalOpen}
                onClose={closeModal}
                model={selectedModel}
            />
        </div>
    );
};

export default HomePage;