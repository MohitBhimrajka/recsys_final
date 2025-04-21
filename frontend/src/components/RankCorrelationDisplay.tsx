// frontend/src/components/RankCorrelationDisplay.tsx
import React, { useMemo } from 'react';
import { RecommendationItem, AllModelsRecs } from '../types';
import Tooltip from './Tooltip';
import { FiInfo } from 'react-icons/fi';
import SpearmanRHO from 'spearman-rho';

interface RankCorrelationDisplayProps {
    allRecs: AllModelsRecs | null;
    ensembleRecs: RecommendationItem[] | null;
    modelPairs: string[][];
    k: number;
}

// Helper to get ranks (Ensure safe access)
const getRanks = (
    itemsToRank: string[],
    recs: RecommendationItem[] | undefined,
    k: number // k is used to define the pool, not limit ranking within the pool here
): { [itemId: string]: number } => {
    const ranks: { [itemId: string]: number } = {};
    const scoreMap: { [itemId: string]: number } = {};

    // Get scores for items *present* in the model's recommendations
    recs?.forEach(rec => {
        if (itemsToRank.includes(rec.presentation_id)) {
            scoreMap[rec.presentation_id] = rec.score;
        }
    });

    // Assign ranks based on scores for items within the pool
    // Items not in the model's recs (even within the pool) get lowest rank implicitly
    const sortedItems = itemsToRank.sort((itemA, itemB) => {
        const scoreA = scoreMap[itemA] ?? -Infinity;
        const scoreB = scoreMap[itemB] ?? -Infinity;
        return scoreB - scoreA; // Descending score
    });

    sortedItems.forEach((itemId, index) => {
        ranks[itemId] = index + 1;
    });

    return ranks;
};


const calculateSpearman = (ranks1: number[], ranks2: number[]): number => {
    if (ranks1.length !== ranks2.length || ranks1.length < 2) return NaN;
    try {
        const rhoCalculator = new SpearmanRHO(ranks1, ranks2);
        const rho = rhoCalculator.calc();
        // Handle potential non-numeric results from the library
        return typeof rho === 'number' && !isNaN(rho) ? rho : NaN;
    } catch (e) { console.error("Spearman calc error:", e); return NaN; }
};

const getCorrelationColor = (value: number): string => {
    if (isNaN(value)) return 'text-text-muted italic'; // Style for N/A
    if (value > 0.6) return 'text-green-400';
    if (value > 0.2) return 'text-yellow-400';
    if (value > -0.2) return 'text-text-secondary';
    return 'text-red-400';
};


const RankCorrelationDisplay: React.FC<RankCorrelationDisplayProps> = ({
    allRecs, ensembleRecs, modelPairs, k
}) => {

    const correlations = useMemo(() => {
        // Determine available models based on non-empty recommendations
        const availableModels = new Set<string>();
        if (ensembleRecs && ensembleRecs.length > 0) availableModels.add('ensemble');
        if (allRecs) {
            Object.entries(allRecs).forEach(([name, recs]) => {
                if (recs && recs.length > 0) availableModels.add(name);
            });
        }
        if (availableModels.size < 2) return []; // Need at least two models

        // Define item pool (union of top K from available models)
        const itemPoolSet = new Set<string>();
        availableModels.forEach(modelName => {
             const recs = modelName === 'ensemble' ? ensembleRecs : allRecs?.[modelName];
             recs?.slice(0, k).forEach(r => itemPoolSet.add(r.presentation_id));
        });
        const itemsToRank = Array.from(itemPoolSet);
        if (itemsToRank.length < 2) return []; // Need at least 2 items to correlate

        // Get ranks for each available model
        const modelRanks: { [modelName: string]: { [itemId: string]: number } } = {};
        availableModels.forEach(modelName => {
            const recs = modelName === 'ensemble' ? ensembleRecs : allRecs?.[modelName];
            modelRanks[modelName] = getRanks(itemsToRank, recs ?? undefined, itemsToRank.length);
        });

        // Calculate correlations for specified pairs IF both models are available
        const results: { modelA: string; modelB: string; rho: number }[] = [];
        modelPairs.forEach(([modelA, modelB]) => {
            if (availableModels.has(modelA) && availableModels.has(modelB)) {
                const ranksA = itemsToRank.map(item => modelRanks[modelA][item]);
                const ranksB = itemsToRank.map(item => modelRanks[modelB][item]);
                const rho = calculateSpearman(ranksA, ranksB);
                results.push({ modelA, modelB, rho });
            } else {
                 // Optionally add placeholder if needed, but filtering pairs is cleaner
                 // results.push({ modelA, modelB, rho: NaN });
            }
        });
        return results;

    }, [allRecs, ensembleRecs, modelPairs, k]);

    if (correlations.length === 0) {
        return <p className="text-sm text-text-muted text-center italic py-4">Rank correlation cannot be calculated (requires at least two models with recommendations).</p>;
    }

    return (
        <div className="overflow-x-auto">
            <table className="w-full text-left text-sm border-collapse">
                <thead className="border-b border-border-color">
                    <tr>
                        <th className="p-2 font-semibold text-text-primary">Model Pair</th>
                        <th className="p-2 font-semibold text-text-primary text-right flex items-center justify-end gap-1">
                            Spearman's Rho
                             <Tooltip content={`Correlation between item rankings based on Top ${k} pool (-1 to 1). 1=identical, 0=uncorrelated.`} position="top">
                                <button className='text-text-muted hover:text-primary inline-block align-middle'><FiInfo size={13} /></button>
                            </Tooltip>
                        </th>
                    </tr>
                </thead>
                <tbody>
                    {correlations.map(({ modelA, modelB, rho }, index) => (
                        <tr key={index} className="border-b border-border-color/50 hover:bg-surface/30">
                            <td className="p-2 text-text-secondary whitespace-nowrap">{modelA} vs {modelB}</td>
                            <td className={`p-2 font-medium text-right ${getCorrelationColor(rho)}`}>
                                {isNaN(rho) ? 'N/A' : rho.toFixed(3)}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

export default RankCorrelationDisplay;