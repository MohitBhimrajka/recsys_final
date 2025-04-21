// frontend/src/components/AnalysisDashboard.tsx
import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { RecommendationItem, AllModelsRecs } from '../types';
import { FiBarChart2, FiShare2, FiActivity, FiTarget, FiLink, FiDatabase, FiHelpCircle } from 'react-icons/fi';

// Import Analysis Components
import AnalysisCard from './AnalysisCard';
import ScoreComparisonChart from '../components/ScoreComparisonChart';
import ScoreDistributionChart from '../components/ScoreDistributionChart';
import OverlapChart from '../components/OverlapChart';
import RankComparisonTable from '../components/RankComparisonTable';
import RankCorrelationDisplay from '../components/RankCorrelationDisplay';
import RecommendationConsensus from '../components/RecommendationConsensus';

// Constants for comparisons (moved inside or passed as props if needed)
const COMPARISON_MODELS_RANK_TABLE = ['ensemble', 'ItemCF', 'NCF (e=15)', 'Hybrid (e=15)', 'ALS (f=100)'];
const COMPARISON_MODELS_SCORE_DIST = ['ensemble', 'ItemCF', 'NCF (e=15)', 'Hybrid (e=15)', 'ALS (f=100)', 'Popularity'];
const OVERLAP_MODELS_PAIRS = [ ['ItemCF', 'NCF (e=15)'], ['ItemCF', 'Hybrid (e=15)'], ['NCF (e=15)', 'Hybrid (e=15)'], ['ItemCF', 'ALS (f=100)'] ];
const CORRELATION_MODEL_PAIRS = [ ['ItemCF', 'NCF (e=15)'], ['ItemCF', 'Hybrid (e=15)'], ['ItemCF', 'ALS (f=100)'], ['NCF (e=15)', 'Hybrid (e=15)'] ];
const TOP_K_RANK_TABLE = 5;
const TOP_K_ANALYSIS = 9; // For Overlap, Score Dist, Consensus, Correlation

interface AnalysisDashboardProps {
  allRecs: AllModelsRecs | null;
  ensembleRecs: RecommendationItem[] | null;
  selectedUserId: number | null;
  modelColors: { [key: string]: string };
}

// Helper to safely get Top K recommendations
const getTopKRecs = (recsData: AllModelsRecs | null, ensembleData: RecommendationItem[] | null, modelName: string, k: number): RecommendationItem[] => {
    const modelKey = modelName === 'Combined' ? 'ensemble' : modelName; // Handle combined key
    if (modelKey === 'ensemble') {
        return ensembleData?.slice(0, k) || [];
    }
    return recsData?.[modelKey]?.slice(0, k) || [];
};

const AnalysisDashboard: React.FC<AnalysisDashboardProps> = ({
    allRecs, ensembleRecs, selectedUserId, modelColors
}) => {
    const [selectedComparisonItem, setSelectedComparisonItem] = useState<string | null>(null);

    // Reset item selection when user changes
    React.useEffect(() => {
        setSelectedComparisonItem(null);
    }, [selectedUserId]);

    // --- Memoized Data Preparation --- (Ensure safe access to potentially null recs)

    const availableModels = useMemo(() => {
        const models = new Set<string>();
        if (ensembleRecs && ensembleRecs.length > 0) models.add('Combined');
        if (allRecs) {
            Object.entries(allRecs).forEach(([name, recs]) => {
                if (recs && recs.length > 0) models.add(name);
            });
        }
        return Array.from(models);
    }, [allRecs, ensembleRecs]);

    const uniqueRecommendedItems = useMemo(() => {
        const items = new Set<string>();
        availableModels.forEach(modelName => {
            getTopKRecs(allRecs, ensembleRecs, modelName, TOP_K_ANALYSIS).forEach(r => items.add(r.presentation_id));
        });
        return Array.from(items).sort();
    }, [allRecs, ensembleRecs, availableModels]);

    const scoreComparisonData = useMemo(() => {
        if (!selectedComparisonItem || availableModels.length === 0) return [];
        const data: { model: string; score: number }[] = [];
        availableModels.forEach(modelName => {
            const recs = getTopKRecs(allRecs, ensembleRecs, modelName, TOP_K_ANALYSIS + 5); // Fetch slightly more
            const item = recs.find(r => r.presentation_id === selectedComparisonItem);
            if (item) data.push({ model: modelName, score: item.score });
        });
        return data.sort((a, b) => b.score - a.score);
    }, [selectedComparisonItem, allRecs, ensembleRecs, availableModels]);

    const scoreDistributionData = useMemo(() => {
        const distribution: { model: string; scores: number[] }[] = [];
        const modelsToConsider = COMPARISON_MODELS_SCORE_DIST.filter(m => availableModels.includes(m === 'ensemble' ? 'Combined' : m));
        modelsToConsider.forEach(modelName => {
            const recs = getTopKRecs(allRecs, ensembleRecs, modelName, TOP_K_ANALYSIS);
            if (recs.length > 0) {
                distribution.push({ model: modelName === 'ensemble' ? 'Combined' : modelName, scores: recs.map(r => r.score) });
            }
        });
        return distribution;
    }, [allRecs, ensembleRecs, availableModels]);

    const overlapData = useMemo(() => {
        return OVERLAP_MODELS_PAIRS
            .filter(([m1, m2]) => availableModels.includes(m1) && availableModels.includes(m2)) // Only compare available models
            .map(([modelA, modelB]) => {
                const recsA_ids = new Set(getTopKRecs(allRecs, ensembleRecs, modelA, TOP_K_ANALYSIS).map(r => r.presentation_id));
                const recsB_ids = new Set(getTopKRecs(allRecs, ensembleRecs, modelB, TOP_K_ANALYSIS).map(r => r.presentation_id));
                const overlapCount = Array.from(recsA_ids).filter(item => recsB_ids.has(item)).length;
                return { pair: `${modelA} vs ${modelB}`, overlap: overlapCount, modelA, modelB };
            });
    }, [allRecs, ensembleRecs, availableModels]);

    // --- Check for sufficient data ---
    const hasSufficientData = ensembleRecs !== null || allRecs !== null;

    if (!selectedUserId) {
        return <p className="text-center text-text-muted italic py-10">Select a student to view analysis.</p>;
    }
    if (!hasSufficientData) {
         return (
            <div className='text-center py-10'>
                 <FiDatabase size={30} className="mx-auto text-text-muted mb-2"/>
                 <p className="text-text-muted italic">Recommendation data is currently unavailable for student {selectedUserId}.</p>
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
            {/* Header */}
            <div className="text-center">
                <motion.h2
                    className="text-2xl md:text-3xl font-semibold mb-2 text-text-primary"
                    initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
                >
                    Analysis & Comparison
                </motion.h2>
                <motion.p
                    className="text-sm text-text-muted max-w-xl mx-auto"
                     initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.1 }}
                >
                    Comparing model behaviors for student <span className="font-semibold text-text-secondary">{selectedUserId}</span>.
                </motion.p>
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
                    description="Direct comparison of the highest-ranked course presentations."
                    isEmpty={!ensembleRecs && !allRecs} // Check if any data exists
                >
                     <RankComparisonTable
                         allRecs={allRecs}
                         ensembleRecs={ensembleRecs}
                         k={TOP_K_RANK_TABLE}
                         modelsToCompare={COMPARISON_MODELS_RANK_TABLE} // Pass only the models intended for this table
                     />
                </AnalysisCard>

                 {/* --- Recommendation Overlap --- */}
                <AnalysisCard
                    title={`Top ${TOP_K_ANALYSIS} Recommendation Overlap`}
                    icon={<FiShare2 size={20}/>}
                    tooltipContent={`Number of shared recommendations within the top ${TOP_K_ANALYSIS} between pairs of key models.`}
                    description="Measures similarity in recommendations between model pairs."
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
                        <div className='h-48 sm:h-full min-h-[150px]'> {/* Ensure chart has height */}
                            <OverlapChart data={overlapData} modelColors={modelColors} />
                        </div>
                    </div>
                </AnalysisCard>

                {/* --- Item-Specific Score Comparison --- */}
                 <AnalysisCard
                    title="Score Comparison per Item"
                    icon={<FiActivity size={20}/>}
                    tooltipContent="Select a recommended item to see how its predicted relevance score varies across the different models."
                    description="Visualize how models rate the same course presentation."
                    isEmpty={uniqueRecommendedItems.length === 0}
                    emptyMessage="No common recommendations found to compare scores."
                >
                    <select
                         value={selectedComparisonItem || ''}
                         onChange={(e) => setSelectedComparisonItem(e.target.value || null)}
                         className="w-full p-2 bg-background border border-border-color rounded-md text-text-secondary text-sm focus:ring-1 focus:ring-primary focus:border-primary mb-4 disabled:opacity-50"
                         disabled={uniqueRecommendedItems.length === 0}
                     >
                         <option value="">-- Select Presentation to Compare Scores --</option>
                         {uniqueRecommendedItems.map(itemId => (
                             <option key={itemId} value={itemId}>{itemId}</option>
                         ))}
                     </select>
                     <ScoreComparisonChart data={scoreComparisonData} modelColors={modelColors} />
                     {!selectedComparisonItem && uniqueRecommendedItems.length > 0 && <p className="text-xs text-text-muted text-center italic pt-4">Select an item above to see the score chart.</p>}
                 </AnalysisCard>

                 {/* --- Score Distribution --- */}
                  <AnalysisCard
                    title={`Top ${TOP_K_ANALYSIS} Score Distribution`}
                    icon={<FiBarChart2 size={20} transform="rotate(90)"/>}
                    tooltipContent={`Shows the distribution (min, max, median, quartiles) of predicted scores for the top ${TOP_K_ANALYSIS} recommendations from each available model.`}
                    description="Understand the range and central tendency of scores per model."
                    isEmpty={scoreDistributionData.length === 0}
                >
                    <ScoreDistributionChart data={scoreDistributionData} modelColors={modelColors} />
                 </AnalysisCard>

                 {/* --- Rank Correlation --- */}
                 <AnalysisCard
                    title="Model Rank Correlation"
                    icon={<FiLink size={20}/>}
                    tooltipContent={`Spearman's Rho measures how similar the ranking order is between model pairs (based on Top ${TOP_K_ANALYSIS} pool). 1 = identical ranking, 0 = no correlation.`}
                    description="Quantify the similarity in ranking behavior between models."
                    isEmpty={!allRecs && !ensembleRecs} // Simplified check, component handles internal logic
                >
                     <RankCorrelationDisplay
                         allRecs={allRecs}
                         ensembleRecs={ensembleRecs}
                         modelPairs={CORRELATION_MODEL_PAIRS}
                         k={TOP_K_ANALYSIS}
                     />
                 </AnalysisCard>

                 {/* --- Consensus & Diversity --- */}
                 <AnalysisCard
                    title={`Top ${TOP_K_RANK_TABLE} Consensus & Diversity`}
                    icon={<FiTarget size={20}/>}
                    tooltipContent={`Highlights items recommended by multiple models (Consensus) versus those suggested uniquely by specific models (Diversity) within the Top ${TOP_K_RANK_TABLE}.`}
                    description="Identify agreement and unique suggestions among models."
                    isEmpty={!allRecs && !ensembleRecs} // Simplified check
                 >
                    <RecommendationConsensus
                        allRecs={allRecs}
                        ensembleRecs={ensembleRecs}
                        modelsToCompare={COMPARISON_MODELS_RANK_TABLE.filter(m => availableModels.includes(m === 'ensemble' ? 'Combined' : m))} // Filter models for consensus check
                        k={TOP_K_RANK_TABLE}
                    />
                 </AnalysisCard>

            </motion.div>

            {/* Explanatory Footer */}
             <motion.div
                 className="text-center mt-12 text-xs text-text-muted border-t border-border-color/30 pt-6"
                 initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}
            >
                 <FiHelpCircle className="inline mr-1" />
                 This analysis uses the top {TOP_K_ANALYSIS} recommendations for most calculations, and top {TOP_K_RANK_TABLE} for the rank table and consensus. Score distributions and overlaps depend on data availability per model.
             </motion.div>
        </div>
    );
};

export default AnalysisDashboard;