// frontend/src/components/RankComparisonTable.tsx
import React from 'react';
import { RecommendationItem, AllModelsRecs } from '../types';

interface RankComparisonTableProps {
    allRecs: AllModelsRecs | null;
    ensembleRecs: RecommendationItem[] | null;
    k: number;
    modelsToCompare: string[]; // Models to include in columns
}

const RankComparisonTable: React.FC<RankComparisonTableProps> = ({
    allRecs, ensembleRecs, k, modelsToCompare
}) => {
    if (!allRecs && !ensembleRecs) {
        return <p className="text-sm text-text-muted text-center italic">Comparison data not available.</p>;
    }

    const tableData: { rank: number; [modelName: string]: string | number }[] = [];

    // Use provided modelsToCompare list
    const validModels = modelsToCompare.filter(name => name === 'ensemble' || (allRecs && allRecs[name]));

    for (let i = 0; i < k; i++) {
        const row: { rank: number; [modelName: string]: string | number } = { rank: i + 1 };

        validModels.forEach(modelName => {
            let presentationId = '-';
            if (modelName === 'ensemble') {
                presentationId = ensembleRecs?.[i]?.presentation_id || '-';
            } else if (allRecs && allRecs[modelName]) {
                 presentationId = allRecs[modelName]?.[i]?.presentation_id || '-';
            }
            // Use 'Combined' for display label if 'ensemble'
            row[modelName === 'ensemble' ? 'Combined' : modelName] = presentationId;
        });
        tableData.push(row);
    }

    // Get final display column names (replace 'ensemble' with 'Combined')
    const displayColumns = validModels.map(name => name === 'ensemble' ? 'Combined' : name);

    return (
        <div className="overflow-x-auto">
            <table className="w-full text-left text-sm border-collapse">
                <thead className="border-b border-border-color">
                    <tr>
                        <th className="p-2 font-semibold text-text-primary">Rank</th>
                        {displayColumns.map(name => (
                            <th key={name} className="p-2 font-semibold text-text-primary whitespace-nowrap">{name}</th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {tableData.map(row => (
                        <tr key={row.rank} className="border-b border-border-color/50 hover:bg-surface/50">
                            <td className="p-2 font-semibold text-center text-text-secondary">{row.rank}</td>
                            {displayColumns.map(name => (
                                <td key={name} className="p-2 font-mono text-xs text-text-muted" title={row[name] !== '-' ? String(row[name]) : ''}>
                                    {row[name]}
                                </td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

export default RankComparisonTable;