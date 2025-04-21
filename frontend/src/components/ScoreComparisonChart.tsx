// frontend/src/components/ScoreComparisonChart.tsx
import React from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Cell } from 'recharts';

interface ScoreComparisonDataPoint {
    model: string;
    score: number;
}

interface ScoreComparisonChartProps {
    data: ScoreComparisonDataPoint[];
    modelColors: { [key: string]: string };
}

const ScoreComparisonChart: React.FC<ScoreComparisonChartProps> = ({ data, modelColors }) => {
    if (!data || data.length === 0) {
        return <p className="text-sm text-text-muted text-center italic py-4">Select an item above to compare scores.</p>;
    }
    // Sort data for consistent bar order if needed, or use as is
    const sortedData = data.sort((a, b) => b.score - a.score); // Example: sort by score descending

    // Determine Y-axis domain based on score range
    const scores = data.map(d => d.score);
    const yMin = Math.min(...scores, 0); // Ensure domain starts at 0 or below if scores are negative
    const yMax = Math.max(...scores);
    const yPadding = Math.abs(yMax - yMin) * 0.1; // 10% padding
    const yDomain: [number | string, number | string] = [
        Math.floor(yMin - yPadding), // Ensure lowest value is visible
        Math.ceil(yMax + yPadding) // Ensure highest value is visible
    ];


    return (
        <ResponsiveContainer width="100%" height={250}>
            <BarChart data={sortedData} margin={{ top: 5, right: 5, left: -25, bottom: 5 }} barSize={30}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-color)" opacity={0.3} vertical={false}/>
                <XAxis dataKey="model" tick={{ fontSize: 10, fill: 'var(--color-text-muted)' }} interval={0} />
                <YAxis tick={{ fontSize: 10, fill: 'var(--color-text-muted)' }} domain={yDomain} allowDecimals={true} />
                <RechartsTooltip
                    cursor={{ fill: 'rgba(var(--color-border-color-rgb), 0.1)' }}
                    contentStyle={{ backgroundColor: 'var(--color-surface)', border: '1px solid var(--color-border-color)', borderRadius: '4px', fontSize: '12px' }}
                    labelStyle={{ color: 'var(--color-text-primary)', fontWeight: 'bold' }}
                    itemStyle={{ color: 'var(--color-text-secondary)' }}
                    formatter={(value: number) => [value.toFixed(4), 'Score']} // Format score
                />
                <Bar dataKey="score" name="Predicted Score" radius={[4, 4, 0, 0]}>
                     {sortedData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={modelColors[entry.model] || '#8884d8'} fillOpacity={0.8} />
                     ))}
                 </Bar>
            </BarChart>
        </ResponsiveContainer>
    );
};

export default ScoreComparisonChart;