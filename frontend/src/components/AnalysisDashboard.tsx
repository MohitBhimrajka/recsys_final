// frontend/src/components/AnalysisDashboard.tsx
import React, { useMemo, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { RecommendationItem, AllModelsRecs } from '../types';
import { FiBarChart2, FiShare2, FiTarget, FiLink, FiDatabase, FiHelpCircle, FiAlertTriangle } from 'react-icons/fi';

// Import Analysis Components
import AnalysisCard from './AnalysisCard';
import OverlapChart from './OverlapChart'; // Keep this
import RankComparisonTable from './RankComparisonTable'; // Keep this
import RankCorrelationDisplay from './RankCorrelationDisplay'; // Keep this
import RecommendationConsensus from './RecommendationConsensus'; // Keep this

// Constants for comparisons
const COMPARISON_MODELS_RANK_TABLE = ['ensemble', 'ItemCF', 'NCF (e=15)', 'Hybrid (e=15)', 'ALS (f=100)'];
const OVERLAP_MODELS_PAIRS = [ ['ItemCF', 'NCF (e=15)'], ['ItemCF', 'Hybrid (e=15)'], ['NCF (e=15)', 'Hybrid (e=15)'], ['ItemCF', 'ALS (f=100)'] ];
const CORRELATION_MODEL_PAIRS = [ ['ItemCF', 'NCF (e=15)'], ['ItemCF', 'Hybrid (e=15)'], ['ItemCF', 'ALS (f=100)'], ['NCF (e=15)', 'Hybrid (e=15)'] ];
const TOP_K_RANK_TABLE = 5;
const TOP_K_ANALYSIS = 9; // For Overlap, Consensus, Correlation

interface AnalysisDashboardProps {
  allRecs: AllModelsRecs | null;
  ensembleRecs: RecommendationItem[] | null;
  selectedUserId: number | null;
  modelColors: { [key: string]: string };
}

// Helper to safely get Top K recommendations
const getTopKRecs = (recsData: AllModelsRecs | null, ensembleData: RecommendationItem[] | null, modelName: string, k: number): RecommendationItem[] => {
    const modelKey = modelName === 'Combined' ? 'ensemble' : modelName;
    if (modelKey === 'ensemble') {
        return ensembleData?.slice(0, k) || [];
    }
    return recsData && recsData[modelKey] ? recsData[modelKey].slice(0, k) : [];
};

const AnalysisDashboard: React.FC<AnalysisDashboardProps> = ({
    allRecs, ensembleRecs, selectedUserId, modelColors
}) => {
    const [errorMessages, setErrorMessages] = useState<string[]>([]);

    // Clear errors when user changes
    useEffect(() => {
        setErrorMessages([]);
    }, [selectedUserId]);

    // --- Memoized Data Preparation ---

    const availableModels = useMemo(() => {
        const models = new Set<string>();
        try {
            if (ensembleRecs && ensembleRecs.length > 0) models.add('Combined');
            if (allRecs) {
                Object.entries(allRecs).forEach(([name, recs]) => {
                    if (Array.isArray(recs) && recs.length > 0) models.add(name);
                });
            }
        } catch (e) {
             console.error("Error determining available models:", e);
             setErrorMessages(prev => [...prev, "Failed to determine available models."]);
        }
        return Array.from(models);
    }, [allRecs, ensembleRecs]);

    const overlapData = useMemo(() => {
         try {
            // Filter pairs where *both* models are available
            const validPairs = OVERLAP_MODELS_PAIRS
                .filter(([m1, m2]) => availableModels.includes(m1) && availableModels.includes(m2));

            return validPairs.map(([modelA, modelB]) => {
                const recsA_ids = new Set(getTopKRecs(allRecs, ensembleRecs, modelA, TOP_K_ANALYSIS).map(r => r?.presentation_id).filter(Boolean));
                const recsB_ids = new Set(getTopKRecs(allRecs, ensembleRecs, modelB, TOP_K_ANALYSIS).map(r => r?.presentation_id).filter(Boolean));
                const overlapCount = Array.from(recsA_ids).filter(item => recsB_ids.has(item)).length;
                return { pair: `${modelA} vs ${modelB}`, overlap: overlapCount, modelA, modelB };
            });
         } catch (e) {
              console.error("Error calculating overlap data:", e);
              setErrorMessages(prev => [...prev, "Failed to calculate overlap data."]);
              return [];
         }
    }, [allRecs, ensembleRecs, availableModels]);

    // --- Check for sufficient data ---
    const hasAnyData = availableModels.length > 0;

    if (!selectedUserId) {
        return <p className="text-center text-text-muted italic py-10">Select a student to view analysis.</p>;
    }
    if (!hasAnyData) {
         return (
            <div className='text-center py-10'>
                 <FiDatabase size={30} className="mx-auto text-text-muted mb-2"/>
                 <p className="text-text-muted italic">No recommendation data found for student {selectedUserId} to analyze.</p>
            </div>
         );
    }

    // --- Animation Variants ---
    const containerVariant = {
        hidden: { opacity: 0 },
        visible: { opacity: 1, transition: { staggerChildren: 0.08, delayChildren: 0.1 } }
    };


    // --- Render Dashboard ---
    return (
        <div className="space-y-10 md:space-y-12">
            {/* Header and Introduction */}
            <div className="text-center">
                <motion.h2
                    className="text-2xl md:text-3xl font-semibold mb-3 text-text-primary"
                    initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
                >
                    Analysis & Comparison Dashboard
                </motion.h2>
                <motion.p
                    className="text-sm text-text-muted max-w-2xl mx-auto mb-6"
                     initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.1 }}
                >
                    Explore how different algorithms recommend courses for student <span className="font-semibold text-text-secondary">{selectedUserId}</span>. This helps understand model agreement, ranking similarities, and potential diversity in suggestions.
                </motion.p>
                 {/* Display Errors */}
                 {errorMessages.length > 0 && (
                     <motion.div
                         initial={{ opacity: 0}} animate={{ opacity: 1 }}
                         className="mt-4 max-w-lg mx-auto bg-red-900/30 border border-red-600/50 text-red-200 text-xs p-3 rounded-md text-left"
                     >
                         <h4 className="font-semibold mb-1 flex items-center gap-1"><FiAlertTriangle/> Analysis Errors:</h4>
                         <ul className="list-disc list-inside">
                             {errorMessages.map((err, i) => <li key={i}>{err}</li>)}
                         </ul>
                         <p className="mt-2 text-red-300/80">Some analysis sections might be incomplete.</p>
                     </motion.div>
                 )}
            </div>

            {/* Grid for Analysis Sections */}
            <motion.div
                className="grid grid-cols-1 lg:grid-cols-2 gap-6 md:gap-8 items-start"
                variants={containerVariant}
                initial="hidden"
                animate="visible"
            >
                {/* --- Rank Comparison Table --- */}
                <AnalysisCard
                    title={`Top ${TOP_K_RANK_TABLE} Rank Comparison`}
                    icon={<FiBarChart2 size={20}/>}
                    tooltipContent="Shows the top items suggested by key models side-by-side. Helps identify consensus and divergence in rankings."
                    description={`Compares the highest-ranked ${TOP_K_RANK_TABLE} course presentations across selected models.`}
                    isEmpty={!hasAnyData}
                >
                     <RankComparisonTable
                         allRecs={allRecs}
                         ensembleRecs={ensembleRecs}
                         k={TOP_K_RANK_TABLE}
                         // Filter models *before* passing to the table component
                         modelsToCompare={COMPARISON_MODELS_RANK_TABLE.filter(m => availableModels.includes(m === 'ensemble' ? 'Combined' : m))}
                     />
                     {/* Add textual summary example */}
                     <p className="text-xs text-text-muted mt-3 pt-3 border-t border-border-color/30">
                        Look for presentations consistently ranked highly (agreement) or those appearing only for certain models (disagreement).
                     </p>
                </AnalysisCard>

                 {/* --- Recommendation Overlap --- */}
                <AnalysisCard
                    title={`Top ${TOP_K_ANALYSIS} Recommendation Overlap`}
                    icon={<FiShare2 size={20}/>}
                    tooltipContent={`Number of shared recommendations within the top ${TOP_K_ANALYSIS} between pairs of key models.`}
                    description="Quantifies how many suggestions are identical between model pairs."
                    isEmpty={overlapData.length === 0}
                    emptyMessage="Not enough comparable model data available."
                >
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-4 items-center">
                        {/* Text Summary */}
                        <div className="text-sm space-y-1.5 pr-2 border-r border-border-color/30 sm:border-r-0 sm:pr-0">
                            {overlapData.map(({ pair, overlap }, index) => (
                                <p key={index} className="flex items-center justify-between text-text-muted">
                                    <span>{pair}</span>
                                    <span className="font-medium text-text-secondary bg-background/50 px-1.5 rounded">{overlap} / {TOP_K_ANALYSIS}</span>
                                </p>
                            ))}
                             {overlapData.length === 0 && <p className='text-xs italic text-center'>No overlap data to display.</p>}
                        </div>
                        {/* Chart */}
                        <div className='h-48 sm:h-full min-h-[150px]'>
                            <OverlapChart data={overlapData} modelColors={modelColors} />
                        </div>
                    </div>
                     <p className="text-xs text-text-muted mt-3 pt-3 border-t border-border-color/30">
                        Higher overlap suggests models have similar views on relevant items for this user. Low overlap indicates differing suggestions.
                     </p>
                </AnalysisCard>

                 {/* --- Rank Correlation --- */}
                 <AnalysisCard
                    title="Model Rank Correlation"
                    icon={<FiLink size={20}/>}
                    tooltipContent={`Spearman's Rho measures how similar the ranking order is between model pairs (based on Top ${TOP_K_ANALYSIS} pool). 1 = identical ranking, 0 = no correlation.`}
                    description={`Evaluates similarity in overall ranking preferences using Spearman's Rho (Top ${TOP_K_ANALYSIS}).`}
                    isEmpty={!hasAnyData} // Check general data availability
                >
                     <RankCorrelationDisplay
                         allRecs={allRecs}
                         ensembleRecs={ensembleRecs}
                         modelPairs={CORRELATION_MODEL_PAIRS}
                         k={TOP_K_ANALYSIS}
                     />
                     {/* Add textual summary example */}
                     <p className="text-xs text-text-muted mt-3 pt-3 border-t border-border-color/30">
                        Values close to 1 indicate models rank items very similarly; values near 0 suggest different ranking logic.
                     </p>
                 </AnalysisCard>

                 {/* --- Consensus & Diversity --- */}
                 <AnalysisCard
                    title={`Top ${TOP_K_RANK_TABLE} Consensus & Diversity`}
                    icon={<FiTarget size={20}/>}
                    tooltipContent={`Highlights items recommended by multiple models (Consensus) versus those suggested uniquely by specific models (Diversity) within the Top ${TOP_K_RANK_TABLE}.`}
                    description={`Identifies items with strong agreement vs. unique model suggestions (Top ${TOP_K_RANK_TABLE}).`}
                    isEmpty={!hasAnyData} // Check general data availability
                 >
                    <RecommendationConsensus
                        allRecs={allRecs}
                        ensembleRecs={ensembleRecs}
                        modelsToCompare={COMPARISON_MODELS_RANK_TABLE.filter(m => availableModels.includes(m === 'ensemble' ? 'Combined' : m))}
                        k={TOP_K_RANK_TABLE}
                    />
                    {/* Add textual summary example */}
                    <p className="text-xs text-text-muted mt-3 pt-3 border-t border-border-color/30">
                        High consensus items are likely safe bets. Unique suggestions might offer novelty or cater to specific model strengths.
                     </p>
                 </AnalysisCard>

            </motion.div>

            {/* Explanatory Footer */}
             <motion.div
                 className="text-center mt-12 text-xs text-text-muted border-t border-border-color/30 pt-6"
                 initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}
            >
                 <FiHelpCircle className="inline mr-1" />
                 Analysis considers Top {TOP_K_ANALYSIS} items for overlap/correlation and Top {TOP_K_RANK_TABLE} for rank/consensus, using only models that provided results.
             </motion.div>
        </div>
    );
};

export default AnalysisDashboard;