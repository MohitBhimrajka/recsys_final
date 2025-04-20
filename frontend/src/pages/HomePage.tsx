// frontend/src/pages/HomePage.tsx
import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FiArrowRight, FiDatabase, FiSettings, FiZap, FiCheckCircle, FiInfo, FiGithub, FiBarChart2 } from 'react-icons/fi';
import ModelInfoModal from '../components/ModelInfoModal';
import { modelInfos, ModelInfo } from '../data/modelInfo';

// Reusable Feature Card Component
const FeatureCard: React.FC<{ icon: React.ReactNode; title: string; children: React.ReactNode }> = ({ icon, title, children }) => (
  <motion.div
    className="bg-surface p-6 rounded-xl shadow-xl border border-border-color text-center h-full flex flex-col transform transition duration-300 hover:scale-[1.03] hover:shadow-primary/20 hover:border-primary/30"
    initial={{ opacity: 0, y: 30 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true, amount: 0.2 }}
    transition={{ duration: 0.5, ease: 'easeOut' }}
  >
    <div className="flex-grow">
        <div className="text-primary text-4xl mb-5 inline-block">{icon}</div>
        <h3 className="text-xl font-semibold text-text-primary mb-3">{title}</h3>
        <p className="text-sm text-text-muted">{children}</p>
    </div>
  </motion.div>
);

// Model Tag Component
const ModelTag: React.FC<{ model: ModelInfo; onClick: () => void; isDemoModel?: boolean }> = ({ model, onClick, isDemoModel = false }) => {
  const baseClasses = "border px-4 py-1.5 rounded-full text-sm cursor-pointer transition-all duration-200 flex items-center gap-1.5";
  const activeClasses = "bg-primary/20 border-primary text-primary font-medium hover:bg-primary/30 shadow-sm";
  const inactiveClasses = "bg-background border-border-color text-text-muted hover:border-primary/50 hover:text-text-secondary";

  return (
    <motion.button
      onClick={onClick}
      className={`${baseClasses} ${isDemoModel ? activeClasses : inactiveClasses}`}
      whileHover={{ y: -2, scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
    >
      {model.name}
       {isDemoModel && <FiCheckCircle size={14} className="flex-shrink-0"/>}
       {!isDemoModel && <FiInfo size={14} className="opacity-50 flex-shrink-0"/>}
    </motion.button>
  );
};

const HomePage: React.FC = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null);

  const openModal = (model: ModelInfo) => {
    setSelectedModel(model);
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setTimeout(() => setSelectedModel(null), 300);
  };

  // --- Framer Motion Variants ---
  const heroVariant = {
    hidden: { opacity: 0, y: -20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.8, delay: 0.2, ease: "easeOut" } },
  };
  const buttonGroupVariant = {
      hidden: { opacity: 0 },
      visible: { opacity: 1, transition: { staggerChildren: 0.15, delayChildren: 0.5 } }
  };
  const buttonVariant = {
      hidden: { opacity: 0, y: 10 },
      visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: 'easeOut' } }
  };
  const sectionVariant = {
     hidden: { opacity: 0, y: 30 },
     visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: "easeOut"} },
  };

  return (
    <div className="text-text-secondary">
      {/* --- Hero Section --- */}
      <section
        className="min-h-screen flex flex-col justify-center items-center text-center px-4 relative overflow-hidden bg-gradient-to-br from-black via-gray-900 to-surface"
      >
         <div className="absolute inset-0 opacity-[0.03] bg-[linear-gradient(to_right,#ffffff12_1px,transparent_1px),linear-gradient(to_bottom,#ffffff12_1px,transparent_1px)] bg-[size:30px_30px]"></div>
        <motion.div variants={heroVariant} initial="hidden" animate="visible" className="z-10">
          <h1 className="text-5xl sm:text-6xl lg:text-7xl font-extrabold text-text-primary mb-5 !leading-tight tracking-tight">
            Unlock Learning Paths
          </h1>
          <p className="text-lg md:text-xl text-text-secondary max-w-3xl mx-auto mb-10">
            Explore personalized course recommendations generated from real Open University student interaction data using collaborative filtering.
          </p>
          <motion.div
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
            variants={buttonGroupVariant}
            initial="hidden"
            animate="visible"
           >
              <motion.div variants={buttonVariant} whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <Link to="/demo" className="btn btn-primary text-lg px-10 py-4 shadow-primary/40 w-full sm:w-auto">
                      Launch Demo <FiArrowRight className="inline ml-2" />
                  </Link>
              </motion.div>
               <motion.div variants={buttonVariant} whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                 <a
                    href="https://github.com/mohitbhimrajka/recsys_final"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn btn-secondary text-lg px-10 py-4 w-full sm:w-auto"
                   >
                     <FiGithub className="inline mr-2"/> View Code
                  </a>
              </motion.div>
          </motion.div>
        </motion.div>
          <motion.div
              className="absolute bottom-10 text-text-muted text-xs animate-bounce z-10"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 1, delay: 1.5 }}
            >
             Scroll Down
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
          <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-text-primary mb-12 text-center">Core Approach: Finding Connections</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 items-stretch">
            <FeatureCard icon={<FiDatabase />} title="Data Processing">
              Raw clickstream data is cleaned, filtered, and aggregated to quantify student engagement (implicit feedback) with each course presentation.
            </FeatureCard>
             <FeatureCard icon={<FiSettings />} title="Item-Based CF Model">
              The system calculates similarity between courses based on user interaction patterns. Courses engaged with by similar student groups are deemed similar.
            </FeatureCard>
             <FeatureCard icon={<FiZap />} title="Personalized Ranking">
              For a given student, unseen courses are scored based on their similarity to courses the student previously interacted with, generating personalized recommendations.
            </FeatureCard>
          </div>
           <motion.div
             className="text-center mt-14"
             initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true, amount: 0.5 }} transition={{ delay: 0.3 }}
            >
             <Link to="/about" className="btn btn-outline text-base px-8">
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
             <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-text-primary mb-6">Choosing the Right Model</h2>
             <p className="text-text-secondary md:text-lg max-w-3xl mx-auto mb-10">
                 We evaluated several recommendation techniques. Click on a model type to learn more about its approach, strengths, and weaknesses. The <strong className="text-primary">ItemCF</strong> model powers this demo due to its effectiveness on the OULAD dataset.
             </p>
             <div className="flex flex-wrap justify-center gap-3 md:gap-4">
                {modelInfos.map((model) => (
                    <ModelTag
                        key={model.id}
                        model={model}
                        onClick={() => openModal(model)}
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
             <p className="text-text-secondary md:text-lg max-w-2xl mx-auto mb-8">
                 Ready to explore? Select a student ID in the demo section to view their personalized course recommendations.
             </p>
             <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
               <Link to="/demo" className="btn btn-primary text-lg px-10 py-4 shadow-primary/40">
                 Go to Demo
               </Link>
             </motion.div>
         </motion.section>

      </div>

      <ModelInfoModal
        isOpen={isModalOpen}
        onClose={closeModal}
        model={selectedModel}
      />
    </div>
  );
};

export default HomePage;