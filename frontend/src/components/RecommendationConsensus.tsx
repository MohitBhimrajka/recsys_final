// frontend/src/components/RecommendationConsensus.tsx
import React, { useMemo } from 'react';
import { RecommendationItem, AllModelsRecs } from '../types';
import { FiUsers, FiUser, FiInfo } from 'react-icons/fi'; // Removed FiCheckCircle
import Tooltip from './Tooltip';

interface RecommendationConsensusProps {
    allRecs: AllModelsRecs | null;
    ensembleRecs: RecommendationItem[] | null;
    modelsToCompare: string[];
    k: number;
}

const getTopKRecs = (recsData: AllModelsRecs | null, ensembleData: RecommendationItem[] | null, modelName: string, k: number): RecommendationItem[] => {
    const modelKey = modelName === 'Combined' ? 'ensemble' : modelName;
    if (modelKey === 'ensemble') {
        return ensembleData?.slice(0, k) || [];
    }
    return recsData?.[modelKey]?.slice(0, k) || [];
};

const RecommendationConsensus: React.FC<RecommendationConsensusProps> = ({
    allRecs, ensembleRecs, modelsToCompare, k
}) => {

    const consensusData = useMemo(() => {
        if (!allRecs && !ensembleRecs && (!modelsToCompare || modelsToCompare.length === 0)) {
            return { highConsensus: [], uniqueSuggestions: {}, availableModelsCount: 0 };
        }

        const itemCounts: { [itemId: string]: { count: number; models: string[] } } = {};
        const modelSuggestions: { [modelName: string]: Set<string> } = {};
        let availableModelsCount = 0;

        // Ensure 'Combined' is handled correctly if 'ensemble' is in modelsToCompare
        const modelKeys = modelsToCompare.map(name => name === 'ensemble' ? 'Combined' : name);

        modelKeys.forEach(modelName => {
            const recs = getTopKRecs(allRecs, ensembleRecs, modelName, k);
            if(recs.length > 0) { // Only count models that provided recommendations
                availableModelsCount++;
                const currentModelSet = new Set<string>();
                recs.forEach(rec => {
                    const itemId = rec.presentation_id;
                    currentModelSet.add(itemId);
                    if (!itemCounts[itemId]) itemCounts[itemId] = { count: 0, models: [] };
                    itemCounts[itemId].count += 1;
                    itemCounts[itemId].models.push(modelName); // Store display name
                });
                modelSuggestions[modelName] = currentModelSet;
            } else {
                 modelSuggestions[modelName] = new Set<string>(); // Still add empty set
            }
        });

        // Only calculate threshold based on models that actually had recommendations
        const highConsensusThreshold = availableModelsCount > 1 ? Math.max(2, Math.ceil(availableModelsCount * 0.5)) : 2;
        const highConsensus: { id: string; count: number; models: string[] }[] = [];
        const uniqueSuggestions: { [modelName: string]: string[] } = {};
        modelKeys.forEach(m => uniqueSuggestions[m] = []); // Initialize for all requested models

        Object.entries(itemCounts).forEach(([itemId, data]) => {
            if (data.count >= highConsensusThreshold) {
                highConsensus.push({ id: itemId, count: data.count, models: data.models });
            }
             // Check against modelSuggestions to ensure the model actually suggested it (handles edge cases if k differs)
            if (data.count === 1 && modelSuggestions[data.models[0]]?.has(itemId)) {
                 const suggestingModel = data.models[0];
                 if(uniqueSuggestions[suggestingModel]) {
                     uniqueSuggestions[suggestingModel].push(itemId);
                 }
            }
        });

        highConsensus.sort((a, b) => b.count - a.count);

        return { highConsensus, uniqueSuggestions, availableModelsCount };

    }, [allRecs, ensembleRecs, modelsToCompare, k]);

    const hasUniqueSuggestions = Object.values(consensusData.uniqueSuggestions).some(list => list.length > 0);
    const consensusThreshold = Math.max(2, Math.ceil(consensusData.availableModelsCount * 0.5));

     if (consensusData.availableModelsCount === 0) {
        return <p className="text-sm text-text-muted text-center italic py-4">No recommendations available to analyze consensus.</p>;
     }


    return (
        <div className="space-y-6 text-sm">
            {/* High Consensus Items */}
            <div>
                <h4 className="font-semibold text-text-secondary mb-2 flex items-center gap-1.5">
                    <FiUsers className="text-green-400" /> High Consensus Items
                    <Tooltip content={`Items recommended by ${consensusThreshold} or more models (out of ${consensusData.availableModelsCount} providing recs) in their Top ${k}.`} position="top">
                        <button className='text-text-muted hover:text-primary'><FiInfo size={13} /></button>
                    </Tooltip>
                </h4>
                {consensusData.highConsensus.length > 0 ? (
                    <ul className="list-none space-y-1">
                        {consensusData.highConsensus.map(item => (
                            <li key={item.id} className="flex items-center justify-between bg-surface/30 px-2 py-1 rounded group">
                                <span className="font-mono text-xs text-text-primary">{item.id}</span>
                                <Tooltip content={`Recommended by: ${item.models.join(', ')}`} position="left" delay={100}>
                                    <span className="text-xs bg-green-800/60 group-hover:bg-green-700/80 text-green-300 px-1.5 py-0.5 rounded-full font-medium cursor-help transition-colors">
                                        {item.count}/{consensusData.availableModelsCount} models
                                    </span>
                                </Tooltip>
                            </li>
                        ))}
                    </ul>
                ) : (
                    <p className="text-text-muted italic text-xs">No items found with high recommendation consensus.</p>
                )}
            </div>

            {/* Unique Suggestions */}
            {hasUniqueSuggestions && (
                 <div>
                    <h4 className="font-semibold text-text-secondary mb-2 flex items-center gap-1.5">
                         <FiUser className="text-blue-400" /> Unique Suggestions
                        <Tooltip content={`Items recommended by only one model within their Top ${k}.`} position="top">
                            <button className='text-text-muted hover:text-primary'><FiInfo size={13} /></button>
                        </Tooltip>
                     </h4>
                     <div className="space-y-2">
                         {Object.entries(consensusData.uniqueSuggestions)
                            .filter(([, items]) => items.length > 0) // Only show models with unique suggestions
                            .map(([modelName, items]) => (
                                 <div key={modelName}>
                                     <p className="text-xs font-medium text-text-muted mb-1">{modelName}:</p>
                                     <div className="flex flex-wrap gap-1">
                                         {items.map(itemId => (
                                             <span key={itemId} className="font-mono text-xs bg-background/80 px-1.5 py-0.5 rounded border border-border-color/50 text-text-secondary/90">
                                                 {itemId}
                                             </span>
                                         ))}
                                     </div>
                                 </div>
                             ))}
                     </div>
                 </div>
             )}
             {!hasUniqueSuggestions && consensusData.availableModelsCount > 0 && (
                  <p className="text-text-muted italic text-xs">No unique suggestions found among the top {k}.</p>
             )}
        </div>
    );
};

export default RecommendationConsensus;