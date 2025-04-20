// frontend/src/pages/AboutPage.tsx
import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FiDatabase, FiFilter, FiLink2, FiTrendingUp, FiCheckCircle, FiXCircle, FiInfo, FiCode, FiGithub, FiExternalLink, FiBarChart2 } from 'react-icons/fi';

// Step Component
const ProcessStep: React.FC<{
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
  isLast?: boolean;
}> = ({ icon, title, children, isLast = false }) => {
  return (
    <motion.div
      className="flex relative pb-12 md:pb-16"
      initial={{ opacity: 0, x: -30 }}
      whileInView={{ opacity: 1, x: 0 }}
      viewport={{ once: true, amount: 0.2 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
    >
      {!isLast && ( <div className="absolute left-6 top-6 -bottom-6 w-0.5 bg-border-color opacity-50"></div> )}
      <div className="flex-shrink-0 w-12 h-12 rounded-full bg-primary/10 border-2 border-primary inline-flex items-center justify-center text-primary relative z-10 shadow-md">
        {icon}
      </div>
      <div className="flex-grow pl-6 md:pl-10">
        <h3 className="font-semibold title-font text-xl md:text-2xl text-text-primary mb-2 tracking-wide">{title}</h3>
        <div className="leading-relaxed text-text-muted text-sm md:text-base">{children}</div>
      </div>
    </motion.div>
  );
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
         <ProcessStep icon={<FiDatabase size={24} />} title="1. Data Foundation (OULAD)">
            We start with the Open University Learning Analytics Dataset, focusing on tables detailing student demographics (`studentInfo`),
            course structure (`courses`), registrations (`studentRegistration`), and detailed VLE interactions (`studentVle`).
        </ProcessStep>

        <ProcessStep icon={<FiFilter size={24} />} title="2. Preprocessing & Filtering">
            Raw data is cleaned (handling missing values, correcting types). VLE interactions are filtered to match active registration periods.
            Critically, sparse data is reduced by removing users and items (course presentations) with very few interactions, improving model stability.
        </ProcessStep>

         <ProcessStep icon={<FiTrendingUp size={24} />} title="3. Quantifying Engagement">
            Filtered VLE clicks are aggregated for each student-presentation pair. We calculate an <code className="bg-surface text-primary/80 text-xs px-1.5 py-0.5 rounded mx-0.5 border border-border-color">implicit_feedback</code> score using <code className="bg-surface text-primary/80 text-xs px-1.5 py-0.5 rounded mx-0.5 border border-border-color">log1p(total_clicks)</code>. Higher scores indicate stronger engagement.
        </ProcessStep>

        <ProcessStep icon={<FiLink2 size={24} />} title="4. Learning Item Connections (ItemCF)">
           The core Item-Based Collaborative Filtering model analyzes the student-item interaction matrix (using the implicit feedback scores). It computes the <code className="bg-surface text-primary/80 text-xs px-1.5 py-0.5 rounded mx-0.5 border border-border-color">cosine similarity</code> between pairs of course presentations. Presentations frequently engaged with by the same users are considered similar.
        </ProcessStep>

         <ProcessStep icon={<FiCheckCircle size={24} />} title="5. Generating Personalized Scores">
            When a student ID is selected in the demo:
             <ul className="list-disc list-inside text-sm text-text-muted mt-2 space-y-1 pl-2">
                <li>The system retrieves the presentations the student interacted with during training.</li>
                <li>It looks up the similarity between these 'seen' items and all other 'unseen' candidate items.</li>
                <li>A score is predicted for each unseen item based on the weighted average of similarities to the student's previously engaged items.</li>
             </ul>
        </ProcessStep>

         <ProcessStep icon={<FiBarChart2 size={24} />} title="6. Ranking & Recommendation" isLast={true}>
            Candidate items are ranked in descending order based on their predicted scores. The top-K (currently 9 in the demo) highest-scoring, previously unseen items are displayed as personalized recommendations.
        </ProcessStep>
      </div>

       {/* Code Link Section */}
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
                  <Link to="/code-explorer" className="btn btn-secondary w-full sm:w-auto">
                      Explore Project Structure <FiExternalLink className="inline ml-2" />
                  </Link>
              </motion.div>
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <a href="https://github.com/mohitbhimrajka/recsys_final" target="_blank" rel="noopener noreferrer" className="btn btn-outline w-full sm:w-auto">
                     View on GitHub <FiGithub className="inline ml-2"/>
                  </a>
              </motion.div>
          </div>
       </motion.div>


      {/* Limitations Section */}
      <motion.div
         className="mt-24 p-8 md:p-10 bg-surface rounded-xl shadow-xl border border-border-color max-w-4xl mx-auto"
         initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true, amount: 0.2 }} transition={{ duration: 0.6 }}
      >
          <h2 className="text-2xl md:text-3xl font-bold text-center mb-8 text-text-primary">
            <FiInfo className="inline mr-2 text-primary" /> Important Considerations
          </h2>
          <ul className="list-none space-y-4 text-text-secondary text-sm md:text-base">
             <li className="flex items-start">
                <FiXCircle className="text-red-500 mr-3 mt-1 flex-shrink-0" size={18}/>
                <span><strong className="text-text-primary">Cold Start:</strong> New users or new courses cannot be handled without retraining or alternative strategies.</span>
             </li>
              <li className="flex items-start">
                <FiXCircle className="text-yellow-500 mr-3 mt-1 flex-shrink-0" size={18}/>
                <span><strong className="text-text-primary">Implicit Feedback:</strong> Click counts are a proxy for interest, not a perfect measure of satisfaction or learning.</span>
             </li>
             <li className="flex items-start">
                <FiXCircle className="text-blue-400 mr-3 mt-1 flex-shrink-0" size={18}/>
                <span><strong className="text-text-primary">Data Snapshot:</strong> The demo model is static; a real-world system needs updates.</span>
             </li>
              <li className="flex items-start">
                <FiXCircle className="text-green-500 mr-3 mt-1 flex-shrink-0" size={18}/>
                <span><strong className="text-text-primary">Item Pool:</strong> The filtering process resulted in 22 unique course presentations, which limits the variety of recommendations.</span>
             </li>
          </ul>
       </motion.div>
    </div>
  );
};

export default AboutPage;