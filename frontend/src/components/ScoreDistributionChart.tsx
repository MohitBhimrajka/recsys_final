// frontend/src/components/ScoreDistributionChart.tsx
import React from 'react';
import { ResponsiveContainer, ComposedChart, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, 
    Bar, Rectangle, Line, ReferenceArea, Cell, Legend } from 'recharts';

interface ScoreDistributionDataPoint {
    model: string;
    scores: number[];
}

interface ScoreDistributionChartProps {
    data: ScoreDistributionDataPoint[];
    modelColors: { [key: string]: string };
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length && payload[0].payload) { // Check payload[0].payload exists
    const data = payload[0].payload;
    const boxName = label || 'Distribution'; // Use label (model name)

    // Safely access quartile data, provide defaults if missing
    const quartile = data.y || [NaN, NaN, NaN]; // 'y' holds the scores array in our setup? No, BoxPlot calculates it internally. Payload[0].payload holds calculated values.
    // Recharts Boxplot payload structure: payload[0].payload has { low, high, median, quartile: [q1, median, q3], outliers?, x, y (original scores)}
    const q1 = (data.quartile?.[0] ?? NaN).toFixed(4);
    const median = (data.quartile?.[1] ?? NaN).toFixed(4);
    const q3 = (data.quartile?.[2] ?? NaN).toFixed(4);
    const low = (data.low ?? NaN).toFixed(4);
    const high = (data.high ?? NaN).toFixed(4);

    return (
      <div className="bg-surface border border-border-color p-3 rounded shadow-lg text-xs backdrop-blur-sm">
        <p className="font-bold text-primary mb-1">{boxName}</p>
        <p className='text-text-secondary'>Max: <span className='font-medium'>{high}</span></p>
        <p className='text-text-secondary'>Q3 (75%): <span className='font-medium'>{q3}</span></p>
        <p className='text-text-secondary'>Median: <span className='font-medium'>{median}</span></p>
        <p className='text-text-secondary'>Q1 (25%): <span className='font-medium'>{q1}</span></p>
        <p className='text-text-secondary'>Min: <span className='font-medium'>{low}</span></p>
      </div>
    );
  }
  return null;
};

const ScoreDistributionChart: React.FC<ScoreDistributionChartProps> = ({ data, modelColors }) => {
    if (!data || data.length === 0) {
        return <p className="text-sm text-text-muted text-center italic py-4 h-full flex items-center justify-center">Score distribution data unavailable.</p>;
    }

    // Prepare data for custom box plot
    const chartData = data.map(item => {
        const scores = [...item.scores].sort((a, b) => a - b);
        const min = scores[0];
        const max = scores[scores.length - 1];
        const q1 = scores[Math.floor(scores.length * 0.25)];
        const median = scores[Math.floor(scores.length * 0.5)];
        const q3 = scores[Math.floor(scores.length * 0.75)];
        
        return {
            x: item.model,
            min,
            q1,
            median,
            q3, 
            max,
            quartile: [q1, median, q3],
            low: min,
            high: max,
            y: item.scores // Keep original scores for tooltip
        };
    });

    // Find global min/max across all scores for Y-axis domain
    const allScores = data.flatMap(d => d.scores);
    const yMin = Math.min(...allScores, 0); // Ensure domain includes 0
    const yMax = Math.max(...allScores);
    const yPadding = Math.abs(yMax - yMin) * 0.1; // 10% padding
    const yDomain: [number | string, number | string] = [
         yMin - yPadding < 0 ? Math.floor((yMin - yPadding)*10)/10 : 0, // Adjust floor for neg
         Math.ceil((yMax + yPadding)*10)/10 // Ceil with padding
    ];

    // Custom box plot rectangle
    const CustomBox = (props: any) => {
        const { x, y, width, height, dataKey, datum, fill, stroke } = props;
        const entry = datum;
        
        if (!entry) return null;
        
        // Calculate positions
        const boxWidth = width * 0.6;
        const boxX = x + (width - boxWidth) / 2;
        const minY = y(entry.min);
        const maxY = y(entry.max);
        const q1Y = y(entry.q1);
        const medianY = y(entry.median);
        const q3Y = y(entry.q3);
        
        return (
            <g>
                {/* Min-Max line */}
                <line x1={x + width/2} x2={x + width/2} y1={minY} y2={maxY} 
                    stroke={stroke || '#000'} strokeWidth={1} />
                
                {/* Box (Q1-Q3) */}
                <Rectangle
                    x={boxX}
                    y={q3Y}
                    width={boxWidth}
                    height={q1Y - q3Y}
                    fill={fill || '#8884d8'}
                    stroke={stroke || '#000'}
                    fillOpacity={0.6}
                />
                
                {/* Median line */}
                <line x1={boxX} x2={boxX + boxWidth} y1={medianY} y2={medianY} 
                    stroke={stroke || '#000'} strokeWidth={1.5} />
                
                {/* Min cap */}
                <line x1={boxX + boxWidth * 0.25} x2={boxX + boxWidth * 0.75} y1={minY} y2={minY} 
                    stroke={stroke || '#000'} strokeWidth={1} />
                
                {/* Max cap */}
                <line x1={boxX + boxWidth * 0.25} x2={boxX + boxWidth * 0.75} y1={maxY} y2={maxY} 
                    stroke={stroke || '#000'} strokeWidth={1} />
            </g>
        );
    };

    return (
        // Ensure container has height defined by parent (e.g., AnalysisCard)
        <ResponsiveContainer width="100%" height="100%" minHeight={250}>
            <ComposedChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 45 }}> {/* Increased bottom margin */}
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-color)" opacity={0.3} vertical={false}/>
                <XAxis
                    dataKey="x"
                    tick={{ fontSize: 10, fill: 'var(--color-text-muted)' }}
                    angle={-35} // Angle labels slightly more
                    textAnchor="end"
                    interval={0}
                    height={55} // Allocate more height
                />
                <YAxis tick={{ fontSize: 10, fill: 'var(--color-text-muted)' }} domain={yDomain} allowDecimals={true} width={40}/>
                <RechartsTooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(var(--color-border-color-rgb), 0.1)' }}/>
                
                {/* Custom box plots */}
                {chartData.map((entry, index) => (
                    <ReferenceArea
                        key={`box-${index}`}
                        x1={index - 0.4}
                        x2={index + 0.4}
                        ifOverflow="extendDomain"
                        shape={<CustomBox 
                            y={(val: number) => {
                                // Convert value to y coordinate
                                const yRange = (yDomain[1] as number) - (yDomain[0] as number);
                                const ratio = (val - (yDomain[0] as number)) / yRange;
                                const height = 250 - 45 - 5; // Total height minus margins
                                return 5 + height * (1 - ratio); // Invert because SVG y is top-down
                            }}
                            datum={entry}
                            fill={modelColors[entry.x] || '#8884d8'}
                            stroke={modelColors[entry.x] ? `color-mix(in srgb, ${modelColors[entry.x]} 80%, #000)` : '#555'}
                        />}
                    />
                ))}
            </ComposedChart>
        </ResponsiveContainer>
    );
};

export default ScoreDistributionChart;