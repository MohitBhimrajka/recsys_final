// frontend/src/components/OverlapChart.tsx
import React from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Cell } from 'recharts';

interface OverlapDataPoint {
    pair: string;
    overlap: number;
    modelA: string;
    modelB: string;
}

interface OverlapChartProps {
    data: OverlapDataPoint[];
    modelColors: { [key: string]: string };
}

const OverlapChart: React.FC<OverlapChartProps> = ({ data, modelColors }) => {
    if (!data || data.length === 0) {
        // Adjusted style for empty message within potentially smaller container
        return <div className="flex items-center justify-center h-full"><p className="text-xs text-text-muted text-center italic py-2">Overlap data unavailable.</p></div>;
    }

    const maxOverlap = Math.max(...data.map(d => d.overlap), 0);

    return (
        <ResponsiveContainer width="100%" height="100%">
            {/* Added check for data length before rendering chart */}
            {data.length > 0 ? (
                <BarChart data={data} margin={{ top: 5, right: 5, left: -15, bottom: 5 }} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-color)" opacity={0.2} horizontal={false}/>
                    <XAxis type="number" domain={[0, maxOverlap + 1]} tick={{ fontSize: 9, fill: 'var(--color-text-muted)' }} allowDecimals={false} />
                    <YAxis
                        type="category"
                        dataKey="pair"
                        width={90} // Increased width slightly
                        tick={{ fontSize: 9, fill: 'var(--color-text-muted)' }}
                        interval={0}
                    />
                    <RechartsTooltip
                        cursor={{ fill: 'rgba(var(--color-border-color-rgb), 0.1)' }}
                        contentStyle={{ backgroundColor: 'var(--color-surface)', border: '1px solid var(--color-border-color)', borderRadius: '4px', fontSize: '11px' }}
                        labelStyle={{ color: 'var(--color-text-primary)', fontWeight: 'bold' }}
                        itemStyle={{ color: 'var(--color-text-secondary)' }}
                        formatter={(value: number) => [`${value} items`, 'Overlap']}
                    />
                    <Bar dataKey="overlap" barSize={15} radius={[0, 4, 4, 0]}>
                        {data.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={modelColors[entry.modelA] || 'var(--color-primary)'} fillOpacity={0.7} />
                        ))}
                    </Bar>
                </BarChart>
            ) : (
                // Render the empty message again if filtering somehow resulted in empty data
                <div className="flex items-center justify-center h-full"><p className="text-xs text-text-muted text-center italic py-2">No overlap data to display.</p></div>
            )}
        </ResponsiveContainer>
    );
};

export default OverlapChart;