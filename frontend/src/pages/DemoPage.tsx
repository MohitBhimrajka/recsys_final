// frontend/src/pages/DemoPage.tsx
import React, { useState, useCallback, useRef, useReducer, useMemo } from 'react'; // Added useMemo
import UserSelector from '../components/UserSelector';
import RecommendationList from '../components/RecommendationList';
import ErrorMessage from '../components/ErrorMessage';
import SkeletonCard from '../components/SkeletonCard';
import ModelInfoModal from '../components/ModelInfoModal';
import PresentationDetailModal from '../components/PresentationDetailModal';
import { fetchRecommendations, fetchAllModelRecommendations, fetchRandomUser } from '../services/recommendationService';
import { RecommendationItem, AllModelsRecs, modelInfos, findModelInfoByName, ModelInfo, PresentationDetailInfo } from '../types';
import { motion, AnimatePresence } from 'framer-motion';
import { FiRefreshCw, FiHelpCircle, FiUserPlus, FiArrowDownCircle, FiCheckCircle, FiSearch, FiInfo, FiGrid, FiLayers, FiSliders, FiBarChart2, FiInbox, FiChevronRight, FiCpu, FiShare2 } from 'react-icons/fi'; // Added FiShare2 for overlap
// Import Recharts components
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer, Cell } from 'recharts';

// --- TYPES & CONSTANTS ---
type TabId = 'ensemble' | 'itemcf' | 'ncf' | 'hybrid' | 'als' | 'popularity' | 'analysis';

const TABS_CONFIG: { id: TabId; label: string; icon: React.ReactNode; order: number }[] = [
    { id: 'ensemble' as TabId, label: 'Combined Suggestion', icon: <FiLayers size={16}/>, order: 1 },
    { id: 'analysis' as TabId, label: 'Analysis & Comparison', icon: <FiBarChart2 size={16}/>, order: 2 },
    { id: 'itemcf' as TabId, label: 'ItemCF', icon: <FiGrid size={16}/>, order: 3 },
    { id: 'ncf' as TabId, label: 'NCF', icon: <FiCpu size={16}/>, order: 4 },
    { id: 'hybrid' as TabId, label: 'Hybrid', icon: <FiSliders size={16}/>, order: 5 },
    { id: 'als' as TabId, label: 'ALS', icon: <FiSliders size={16}/>, order: 6 },
    { id: 'popularity' as TabId, label: 'Popularity', icon: <FiInbox size={16}/>, order: 7 },
].sort((a, b) => a.order - b.order);

const COMPARISON_MODELS = ['ensemble', 'ItemCF', 'NCF (e=15)', 'Hybrid (e=15)']; // Models for comparison table/chart
const OVERLAP_MODELS_PAIRS = [ ['ItemCF', 'NCF (e=15)'], ['ItemCF', 'Hybrid (e=15)'], ['NCF (e=15)', 'Hybrid (e=15)'] ]; // Pairs for overlap check
const TOP_K_COMPARISON = 5; // K value for rank comparison and overlap

// Chart Colors (consistent and accessible if possible)
const modelColors: { [key: string]: string } = {
  'ensemble': '#06b6d4', // primary.DEFAULT
  'ItemCF': '#8b5cf6', // secondary.DEFAULT
  'NCF (e=15)': '#10b981', // emerald-500
  'Hybrid (e=15)': '#f59e0b', // amber-500
  'ALS (f=100)': '#ef4444', // red-500
  'Popularity': '#6b7280', // gray-500
};


// Card State Reducer (Unchanged)
type CardStateAction = | { type: 'HIDE'; payload: string } | { type: 'TOGGLE_HIGHLIGHT'; payload: string } | { type: 'RESET' };
interface CardDisplayState { hiddenCards: Set<string>; highlightedCards: Set<string>; }
function cardDisplayReducer(state: CardDisplayState, action: CardStateAction): CardDisplayState { /* ... implementation unchanged ... */
    switch (action.type) {
        case 'HIDE': return { ...state, hiddenCards: new Set(state.hiddenCards).add(action.payload) };
        case 'TOGGLE_HIGHLIGHT': { const newHighlighted = new Set(state.highlightedCards); if (newHighlighted.has(action.payload)) newHighlighted.delete(action.payload); else newHighlighted.add(action.payload); return { ...state, highlightedCards: newHighlighted }; }
        case 'RESET': return { hiddenCards: new Set(), highlightedCards: new Set() };
        default: return state;
    }
}

// --- Tab Component (Enhanced with Highlight) ---
interface TabProps {
    label: string;
    icon?: React.ReactNode;
    isActive: boolean;
    onClick: () => void;
    highlight?: boolean; // <-- New prop
}
const Tab: React.FC<TabProps> = ({ label, icon, isActive, onClick, highlight = false }) => {
    // Base classes remain the same
    const baseClasses = "relative flex-shrink-0 flex items-center gap-2 px-4 sm:px-5 py-3 text-sm font-medium transition-colors duration-200 outline-none focus-visible:ring-1 focus-visible:ring-primary rounded-t-md";

    // Active classes remain the same
    const activeClasses = "text-primary";

    // Inactive classes - differentiate based on 'highlight' prop
    const inactiveBaseClasses = "text-text-muted hover:text-text-secondary";
    const inactiveHighlightClasses = "bg-surface/50 hover:bg-surface/80"; // Subtle background for highlighted inactive tab
    const inactiveNormalClasses = "hover:bg-transparent"; // Default inactive hover

    const inactiveCombinedClasses = `${inactiveBaseClasses} ${highlight && !isActive ? inactiveHighlightClasses : inactiveNormalClasses}`;

    return (
        <motion.button
            onClick={onClick} role="tab" aria-selected={isActive}
            className={`${baseClasses} ${isActive ? activeClasses : inactiveCombinedClasses}`} // Apply conditional inactive style
            whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.98 }}
        >
            {icon} <span className="whitespace-nowrap">{label}</span>
            {/* Animated underline for active tab (Unchanged) */}
            {isActive && ( <motion.div className="absolute bottom-[-1px] left-0 right-0 h-0.5 bg-primary" layoutId="activeTabIndicator" initial={false} transition={{ type: 'spring', stiffness: 300, damping: 25 }} /> )}
        </motion.button>
    );
};

// --- Helper for Score Distribution (Unchanged) ---
const ScoreDistributionInfo: React.FC<{ scores: number[] }> = ({ scores }) => { /* ... implementation unchanged ... */
    if (!scores || scores.length === 0) return null;
    const minScore = Math.min(...scores).toFixed(4);
    const maxScore = Math.max(...scores).toFixed(4);
    const avgScore = (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(4);
    return ( <div className="mt-6 text-center text-xs text-text-muted border-t border-border-color/50 pt-4"> Top {scores.length} Scores - Min: <strong className="text-text-secondary">{minScore}</strong> | Avg: <strong className="text-text-secondary">{avgScore}</strong> | Max: <strong className="text-text-secondary">{maxScore}</strong> </div> );
};

// --- Helper for Rank Comparison Table (Unchanged) ---
const RankComparisonTable: React.FC<{ allRecs: AllModelsRecs | null, ensembleRecs: RecommendationItem[] | null, k: number }> = ({ allRecs, ensembleRecs, k }) => { /* ... implementation unchanged ... */
    if (!allRecs || !ensembleRecs) return <p className="text-sm text-text-muted text-center italic">Comparison data not available.</p>;
    const modelsToCompare = COMPARISON_MODELS;
    const tableData: { rank: number; [modelName: string]: string | number }[] = [];
    for (let i = 0; i < k; i++) {
        const row: { rank: number; [modelName: string]: string | number } = { rank: i + 1 };
        row['ensemble'] = ensembleRecs[i]?.presentation_id || '-';
        for (const modelName of modelsToCompare) { if(modelName === 'ensemble') continue; row[modelName] = allRecs[modelName]?.[i]?.presentation_id || '-'; }
        tableData.push(row);
    }
    return ( <div className="overflow-x-auto"> <table className="w-full text-left text-sm border-collapse"> <thead className="border-b border-border-color"> <tr> <th className="p-2 font-semibold text-text-primary">Rank</th> {modelsToCompare.map(name => ( <th key={name} className="p-2 font-semibold text-text-primary whitespace-nowrap">{name === 'ensemble' ? 'Combined' : name}</th> ))} </tr> </thead> <tbody> {tableData.map(row => ( <tr key={row.rank} className="border-b border-border-color/50 hover:bg-surface/50"> <td className="p-2 font-semibold text-center text-text-secondary">{row.rank}</td> {modelsToCompare.map(name => ( <td key={name} className="p-2 font-mono text-xs text-text-muted" title={row[name] !== '-' ? String(row[name]) : ''}> {row[name]} </td> ))} </tr> ))} </tbody> </table> </div> );
 };

// --- Helper Chart Component for Score Comparison ---
const ScoreComparisonChart: React.FC<{ data: { model: string; score: number }[] }> = ({ data }) => {
    if (!data || data.length === 0) {
        return <p className="text-sm text-text-muted text-center italic py-4">Select an item above to compare scores.</p>;
    }
    // Sort data for consistent bar order if needed, or use as is
    const sortedData = data.sort((a, b) => b.score - a.score); // Example: sort by score descending

    return (
        <ResponsiveContainer width="100%" height={250}>
            <BarChart data={sortedData} margin={{ top: 5, right: 5, left: -25, bottom: 5 }} barSize={30}> {/* Adjusted margins */}
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-color)" opacity={0.3}/>
                <XAxis dataKey="model" tick={{ fontSize: 10, fill: 'var(--color-text-muted)' }} interval={0} />
                <YAxis tick={{ fontSize: 10, fill: 'var(--color-text-muted)' }} domain={[0, 'auto']} />
                <RechartsTooltip
                    cursor={{ fill: 'rgba(var(--color-border-color-rgb), 0.2)' }}
                    contentStyle={{ backgroundColor: 'var(--color-surface)', border: '1px solid var(--color-border-color)', borderRadius: '4px', fontSize: '12px' }}
                    labelStyle={{ color: 'var(--color-text-primary)', fontWeight: 'bold' }}
                    itemStyle={{ color: 'var(--color-text-secondary)' }}
                />
                {/* Removed Legend as XAxis labels models */}
                {/* <Legend wrapperStyle={{ fontSize: '10px' }} /> */}
                <Bar dataKey="score" name="Predicted Score" fill="#8884d8" radius={[4, 4, 0, 0]}>
                    {/* Use specific colors per model */}
                     {sortedData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={modelColors[entry.model] || '#8884d8'} />
                     ))}
                 </Bar>
            </BarChart>
        </ResponsiveContainer>
    );
};

// --- Helper for Overlap Summary ---
const OverlapSummary: React.FC<{ allRecs: AllModelsRecs | null, k: number }> = ({ allRecs, k }) => {
    if (!allRecs) return null;

    const overlaps = OVERLAP_MODELS_PAIRS.map(([modelA, modelB]) => {
        const recsA = allRecs[modelA]?.slice(0, k).map(r => r.presentation_id) || [];
        const recsB = allRecs[modelB]?.slice(0, k).map(r => r.presentation_id) || [];
        const overlapCount = new Set([...recsA].filter(item => recsB.includes(item))).size;
        return { modelA, modelB, overlapCount };
    });

    return (
        <ul className="space-y-1 text-sm">
             {overlaps.map(({ modelA, modelB, overlapCount }, index) => (
                 <li key={index} className="flex items-center justify-between text-text-muted">
                     <span className='flex items-center gap-1'>
                         <FiShare2 size={13} className="opacity-60"/> {modelA} ↔ {modelB}
                     </span>
                     <span className="font-medium text-text-secondary">{overlapCount} / {k} shared</span>
                 </li>
             ))}
         </ul>
    );
};


// --- DemoPage Component ---
const DemoPage: React.FC = () => {
  // --- State ---
  const [selectedUserId, setSelectedUserId] = useState<number | null>(null);
  const [ensembleRecommendations, setEnsembleRecommendations] = useState<RecommendationItem[] | null>(null);
  const [allModelRecommendations, setAllModelRecommendations] = useState<AllModelsRecs | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isFetchingRandom, setIsFetchingRandom] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<TabId>('ensemble');
  const [cardState, dispatchCardState] = useReducer(cardDisplayReducer, { hiddenCards: new Set<string>(), highlightedCards: new Set<string>() });
  const [isModelInfoModalOpen, setIsModelInfoModalOpen] = useState(false);
  const [currentModelInfo, setCurrentModelInfo] = useState<ModelInfo | null>(null);
  const [isDetailModalOpen, setIsDetailModalOpen] = useState(false);
  const [currentPresentationDetail, setCurrentPresentationDetail] = useState<PresentationDetailInfo | null>(null);
  // State for Analysis Tab
  const [selectedComparisonItem, setSelectedComparisonItem] = useState<string | null>(null);

  // --- Refs ---
  const userSelectorRef = useRef<any>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  // --- Modal Handlers --- (Unchanged)
  const openModelInfoModal = (modelName: string) => { const modelInfo = findModelInfoByName(modelName); if (modelInfo) { setCurrentModelInfo(modelInfo); setIsModelInfoModalOpen(true); }};
  const closeModelInfoModal = () => setIsModelInfoModalOpen(false);
  const openDetailModal = (presentation: PresentationDetailInfo) => { setCurrentPresentationDetail(presentation); setIsDetailModalOpen(true); };
  const closeDetailModal = () => setIsDetailModalOpen(false);

  // --- Data Fetching Logic --- (Reset comparison item on new user)
  const handleUserSelect = useCallback(async (userId: number | null, scrollIntoView: boolean = false) => {
    dispatchCardState({ type: 'RESET' });
    setEnsembleRecommendations(null); setAllModelRecommendations(null); setActiveTab('ensemble');
    setSelectedComparisonItem(null); // Reset comparison item selection

    if (userId === null) { setSelectedUserId(null); setError(null); setIsLoading(false); return; }
    const numericUserId = parseInt(String(userId), 10);
    if (isNaN(numericUserId)) { setError("Invalid User ID selected."); setIsLoading(false); return; }
    console.log("Fetching recommendations for user:", numericUserId);
    setSelectedUserId(numericUserId); setIsLoading(true); setError(null);
    try {
      const [ensembleResult, allModelsResult] = await Promise.allSettled([ fetchRecommendations(numericUserId, 9), fetchAllModelRecommendations(numericUserId, 9) ]);
      let fetchError: string | null = null;
      if (ensembleResult.status === 'fulfilled') setEnsembleRecommendations(ensembleResult.value); else { console.error("Failed ensemble:", ensembleResult.reason); fetchError = `Failed combined suggestions: ${ensembleResult.reason?.message || 'Unknown error'}. `; }
      if (allModelsResult.status === 'fulfilled') setAllModelRecommendations(allModelsResult.value); else { console.error("Failed all models:", allModelsResult.reason); fetchError = (fetchError || "") + `Failed individual model results: ${allModelsResult.reason?.message || 'Unknown error'}.`; }
      if (fetchError) setError(fetchError + " Please check the API server."); else { setError(null); if (scrollIntoView && resultsRef.current) setTimeout(() => { resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }); }, 150); }
    } catch (err: unknown) { console.error("Fetch error:", err); setError("An unexpected error occurred while fetching data."); setEnsembleRecommendations(null); setAllModelRecommendations(null); }
    finally { setIsLoading(false); }
   }, []);

  const handleRandomUser = useCallback(async () => { /* ... implementation unchanged ... */
    dispatchCardState({ type: 'RESET' }); setIsFetchingRandom(true); setError(null);
    setEnsembleRecommendations(null); setAllModelRecommendations(null); setSelectedUserId(null);
    setSelectedComparisonItem(null); // Reset comparison item
    if (userSelectorRef.current?.select) userSelectorRef.current.select.setValue(null, 'clear');
    try { const randomUser = await fetchRandomUser(); if (randomUser) { const randomId = randomUser.student_id; if (userSelectorRef.current?.select) userSelectorRef.current.select.setValue({ value: randomId, label: String(randomId) }, 'select-option'); setTimeout(() => handleUserSelect(randomId, true), 50); } else throw new Error("API did not return a random user."); }
    catch (err: unknown) { console.error("Random user fetch failed:", err); setError(`Error selecting random user: ${err instanceof Error ? err.message : 'Unknown error'}.`); setSelectedUserId(null); if (userSelectorRef.current?.select) userSelectorRef.current.select.clearValue(); }
    finally { setIsFetchingRandom(false); }
  }, [handleUserSelect]);

  // --- Card Interaction Handlers --- (Unchanged)
  const handleHideCard = (presentationId: string) => dispatchCardState({ type: 'HIDE', payload: presentationId });
  const handleHighlightCard = (presentationId: string) => dispatchCardState({ type: 'TOGGLE_HIGHLIGHT', payload: presentationId });
  const handleCardClick = (recommendation: RecommendationItem) => openDetailModal(recommendation);

  // --- Memoized Calculations for Analysis Tab ---
  // Get all unique recommended items across top N for the dropdown
  const uniqueRecommendedItems = useMemo(() => {
      const items = new Set<string>();
      if (ensembleRecommendations) ensembleRecommendations.forEach(r => items.add(r.presentation_id));
      if (allModelRecommendations) {
          Object.values(allModelRecommendations).forEach(recs => recs.forEach(r => items.add(r.presentation_id)));
      }
      return Array.from(items).sort();
  }, [ensembleRecommendations, allModelRecommendations]);

  // Prepare data for the score comparison chart based on selected item
  const scoreComparisonData = useMemo(() => {
      if (!selectedComparisonItem || (!allModelRecommendations && !ensembleRecommendations)) return [];
      const data: { model: string; score: number }[] = [];

      // Add Ensemble score
      const ensembleItem = ensembleRecommendations?.find(r => r.presentation_id === selectedComparisonItem);
      if (ensembleItem) { data.push({ model: 'Combined', score: ensembleItem.score }); } // Use 'Combined' label

      // Add Individual model scores
      if(allModelRecommendations){
           for (const [modelName, recs] of Object.entries(allModelRecommendations)) {
               const item = recs.find(r => r.presentation_id === selectedComparisonItem);
               if (item) { data.push({ model: modelName, score: item.score }); }
           }
      }
      return data;
  }, [selectedComparisonItem, allModelRecommendations, ensembleRecommendations]);


  // --- Animation Variants --- (Unchanged)
  const containerVariant = { hidden: { opacity: 0 }, visible: { opacity: 1, transition: { staggerChildren: 0.1, delayChildren: 0.1 } } };
  const itemVariant = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut'} } };
  const resultsVariant = { hidden: { opacity: 0, transition: { duration: 0.2 } }, visible: { opacity: 1, transition: { duration: 0.4 } }, exit: { opacity: 0, transition: { duration: 0.2 } } };
  const tabContentVariant = { hidden: { opacity: 0, y: 15 }, visible: { opacity: 1, y: 0, transition: { duration: 0.45, ease: "easeOut", delay: 0.1 } }, exit: { opacity: 0, y: -10, transition: { duration: 0.25, ease: "easeIn" } } };

  // --- Render ---
  return (
    <motion.div className="py-16 md:py-20" variants={containerVariant} initial="hidden" animate="visible">
      {/* Header */}
      <motion.header className="text-center mb-12 md:mb-16 px-4" variants={itemVariant}>
        <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold text-text-primary mb-4 tracking-tight"> Course Recommendation Demo </h1>
        <p className="text-text-secondary md:text-lg max-w-3xl mx-auto"> Select a student ID (or try a random one) to view recommendations. We show a <strong className="text-primary">Combined Suggestion</strong> (best overall guess) and results from <strong className="text-primary">Individual Models</strong> for comparison and analysis. </p>
      </motion.header>

      {/* Control Section */}
      <motion.div className="max-w-lg mx-auto mb-16 md:mb-20 px-4" variants={itemVariant}>
        <div className="bg-surface rounded-xl shadow-xl border border-border-color p-6">
           <div className="flex items-center gap-3 mb-4"> <FiSearch className="text-primary text-xl flex-shrink-0" /> <h2 className="text-lg font-semibold text-text-primary">Select a Student</h2> </div>
           <UserSelector ref={userSelectorRef} onUserSelect={(userId) => handleUserSelect(userId, true)} isLoading={isLoading || isFetchingRandom} />
           <div className="flex items-center my-4"> <span className="flex-grow border-t border-border-color opacity-50"></span> <span className="flex-shrink mx-3 text-xs text-text-muted">OR</span> <span className="flex-grow border-t border-border-color opacity-50"></span> </div>
           <motion.button onClick={handleRandomUser} disabled={isFetchingRandom || isLoading} className="btn btn-secondary btn-sm w-full disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2" whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.98 }} > <FiRefreshCw size={16} className={`${isFetchingRandom ? 'animate-spin' : ''}`} /> {isFetchingRandom ? 'Selecting...' : 'Try a Random Student ID'} </motion.button>
        </div>
      </motion.div>

      {/* --- Results Section --- */}
      <div ref={resultsRef} className="min-h-[400px]">
        <AnimatePresence mode="wait">
            {/* Loading State */}
            {isLoading && selectedUserId && ( <motion.div key="loading" variants={resultsVariant} initial="hidden" animate="visible" exit="exit" className="w-full px-4"> <h2 className="text-xl font-semibold mb-10 text-center text-text-muted animate-pulse"> Generating Recommendations for Student <span className='font-bold text-primary'>{selectedUserId}</span>... </h2> <div className="border-b border-border-color mb-8 flex justify-center"> <div className="animate-pulse bg-surface/80 h-12 w-48 rounded-t-lg mr-1"></div> <div className="animate-pulse bg-surface/80 h-12 w-48 rounded-t-lg ml-1"></div> <div className="animate-pulse bg-surface/80 h-12 w-48 rounded-t-lg ml-1"></div> </div> <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-6 md:gap-8"> {Array.from({ length: 6 }).map((_, index) => <SkeletonCard key={`skel-${index}`} />)} </div> </motion.div> )}
             {/* Error State */}
             {!isLoading && error && ( <motion.div key="error" variants={resultsVariant} initial="hidden" animate="visible" exit="exit"> <ErrorMessage message={error} /> </motion.div> )}
              {/* Initial State */}
              {!isLoading && !error && !selectedUserId && ( <motion.div key="initial-prompt" variants={resultsVariant} initial="hidden" animate="visible" exit="exit" className="text-center text-text-muted pt-10 pb-10 min-h-[300px] px-4"> <motion.div initial={{ y: -5 }} animate={{ y: [0, -5, 0], transition: { duration: 1.5, repeat: Infinity, ease: "easeInOut" } }}> <FiArrowDownCircle size={48} className="mx-auto mb-5 text-border-color opacity-60" /> </motion.div> <p className="text-lg font-medium text-text-secondary">Waiting for Input</p> <p className="text-sm mt-1">Select a student above to see recommendations.</p> </motion.div> )}

              {/* --- Results Display State (Tabs) --- */}
             {!isLoading && !error && selectedUserId && (
                 <motion.div key={`results-${selectedUserId}`} variants={resultsVariant} initial="hidden" animate="visible" exit="exit">
                    {/* Overall Explanation Section */}
                    <motion.div className="mb-10 md:mb-12 px-4 py-5 bg-surface/50 border border-border-color rounded-lg max-w-4xl mx-auto text-sm" initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} >
                        <h3 className="font-semibold text-base text-text-primary mb-3 flex items-center gap-2"><FiInfo size={18}/> Understanding the Results for Student <span className="text-primary">{selectedUserId}</span></h3>
                        <ul className="list-none space-y-1.5 text-text-muted pl-1">
                            <li><FiChevronRight className="inline mr-1 opacity-70"/> The <strong className="text-text-secondary">Combined Suggestion</strong> tab shows our best overall recommendations (a weighted mix of models).</li>
                            <li><FiChevronRight className="inline mr-1 opacity-70"/> The <strong className="text-text-secondary">Individual Model</strong> tabs show raw rankings from each specific algorithm for comparison.</li>
                             <li><FiChevronRight className="inline mr-1 opacity-70"/> The <strong className="text-text-secondary">Analysis & Comparison</strong> tab provides tools to directly compare model outputs.</li>
                            <li><FiChevronRight className="inline mr-1 opacity-70"/> <strong className="text-text-secondary">Scores</strong> indicate predicted relevance (higher is better), and <strong className="text-text-secondary">Rank #1</strong> is the top suggestion for each list.</li>
                        </ul>
                    </motion.div>

                     {/* Tab Buttons */}
                      <div className="border-b border-border-color mb-8 md:mb-12 px-4 overflow-x-auto sticky top-16 bg-background/80 backdrop-blur-sm z-40"> {/* Made tabs sticky */}
                           <nav className="-mb-px flex justify-start sm:justify-center space-x-1 sm:space-x-3" aria-label="Tabs">
                               {TABS_CONFIG.map(tab => ( <Tab key={tab.id} label={tab.label} icon={tab.icon} isActive={activeTab === tab.id} onClick={() => setActiveTab(tab.id)} highlight={tab.id === 'analysis'} /> ))}
                           </nav>
                      </div>

                     {/* Tab Content Area */}
                      <div className="px-4">
                           <AnimatePresence mode="wait">
                               {/* Map through tabs and render active content */}
                               {TABS_CONFIG.map(tab => (
                                   activeTab === tab.id && (
                                       <motion.section key={tab.id} variants={tabContentVariant} initial="hidden" animate="visible" exit="exit">
                                           {/* --- Ensemble Tab --- */}
                                           {tab.id === 'ensemble' && (
                                               <div className="space-y-8 md:space-y-10">
                                                    <div className="text-center"> <h2 className="text-2xl font-semibold mb-2 text-text-primary">Combined Suggestion (Ensemble)</h2> <p className="text-sm text-text-muted max-w-xl mx-auto"> Top suggestions based on a weighted average of scores from all models. Interact with cards below (hide/highlight). </p> </div>
                                                    <RecommendationList recommendations={ensembleRecommendations || []} selectedUserId={selectedUserId} hiddenCards={cardState.hiddenCards} highlightedCards={cardState.highlightedCards} onHideCard={handleHideCard} onHighlightCard={handleHighlightCard} onCardClick={handleCardClick} isIndividualModelList={false} />
                                                    <ScoreDistributionInfo scores={ensembleRecommendations?.map(r => r.score) || []} />
                                                </div>
                                           )}
                                           {/* --- Individual Model Tabs --- */}
                                           {['itemcf', 'ncf', 'hybrid', 'als', 'popularity'].includes(tab.id) && (() => {
                                                const modelInfo = findModelInfoByName(tab.label);
                                                const modelRecs = allModelRecommendations ? allModelRecommendations[tab.label] : [];
                                                return (
                                                    <div className="space-y-8 md:space-y-10">
                                                         <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-1">
                                                             <button onClick={() => openModelInfoModal(tab.label)} className="text-xl md:text-2xl font-semibold text-text-primary flex items-center gap-2 hover:text-primary transition-colors outline-none focus-visible:ring-1 focus-visible:ring-primary rounded text-left" title={`Learn about the ${tab.label} model`}> {tab.label} <FiInfo size={18} className="opacity-70"/> </button>
                                                             {modelInfo?.evaluationScore && ( <p className="text-xs text-left sm:text-right text-text-muted italic whitespace-nowrap">Offline NDCG@10: <span className="font-medium text-text-secondary">{modelInfo.evaluationScore.toFixed(4)}</span></p> )}
                                                        </div>
                                                        {modelInfo?.description && ( <p className="text-sm text-text-muted mb-6">{modelInfo.description.split('.')[0]}.</p> )}
                                                         <RecommendationList recommendations={modelRecs || []} selectedUserId={selectedUserId} onCardClick={handleCardClick} isIndividualModelList={true} modelName={tab.label} />
                                                     </div> );
                                           })()}
                                           {/* --- Analysis Tab --- */}
                                           {tab.id === 'analysis' && (
                                                <div className="space-y-10 md:space-y-12">
                                                    <div className="text-center"> <h2 className="text-2xl font-semibold mb-2 text-text-primary">Analysis & Comparison</h2> <p className="text-sm text-text-muted max-w-xl mx-auto"> Tools to compare how different models rank courses for student <span className="font-semibold text-text-secondary">{selectedUserId}</span>. </p> </div>
                                                    {/* Rank Comparison */}
                                                    <div className="bg-surface/50 rounded-lg border border-border-color/50 p-4 md:p-6 shadow-md">
                                                         <h3 className="text-lg font-semibold text-text-primary mb-3">Top {TOP_K_COMPARISON} Recommendation Comparison</h3>
                                                         <p className="text-xs text-text-muted mb-4">Shows the top courses suggested by key models side-by-side. Helps identify consensus and divergence.</p>
                                                         <RankComparisonTable allRecs={allModelRecommendations} ensembleRecs={ensembleRecommendations} k={TOP_K_COMPARISON}/>
                                                    </div>
                                                     {/* Score Comparison */}
                                                     <div className="bg-surface/50 rounded-lg border border-border-color/50 p-4 md:p-6 shadow-md">
                                                         <h3 className="text-lg font-semibold text-text-primary mb-3">Score Comparison per Item</h3>
                                                          <p className="text-xs text-text-muted mb-4">Select a recommended course presentation below to see how its predicted score varies across the different models.</p>
                                                          <select
                                                               value={selectedComparisonItem || ''}
                                                               onChange={(e) => setSelectedComparisonItem(e.target.value || null)}
                                                               className="w-full p-2 bg-background border border-border-color rounded-md text-text-secondary text-sm focus:ring-1 focus:ring-primary focus:border-primary mb-4"
                                                               disabled={uniqueRecommendedItems.length === 0}
                                                           >
                                                               <option value="">-- Select a Presentation --</option>
                                                               {uniqueRecommendedItems.map(itemId => (
                                                                   <option key={itemId} value={itemId}>{itemId}</option>
                                                               ))}
                                                           </select>
                                                           {/* Render chart only if an item is selected */}
                                                           {selectedComparisonItem && <ScoreComparisonChart data={scoreComparisonData} />}
                                                           {!selectedComparisonItem && uniqueRecommendedItems.length > 0 && <p className="text-sm text-text-muted text-center italic py-4">Select an item above to see the score chart.</p>}
                                                           {uniqueRecommendedItems.length === 0 && <p className="text-sm text-text-muted text-center italic py-4">No recommendations available to compare scores.</p>}
                                                     </div>
                                                    {/* Overlap Summary */}
                                                     <div className="bg-surface/50 rounded-lg border border-border-color/50 p-4 md:p-6 shadow-md">
                                                          <h3 className="text-lg font-semibold text-text-primary mb-3">Top {TOP_K_COMPARISON} Recommendation Overlap</h3>
                                                          <p className="text-xs text-text-muted mb-4">Shows how many of the top {TOP_K_COMPARISON} recommendations are shared between key model pairs.</p>
                                                          <OverlapSummary allRecs={allModelRecommendations} k={TOP_K_COMPARISON} />
                                                     </div>
                                                </div>
                                           )}
                                       </motion.section>
                                   )
                               ))}
                           </AnimatePresence>
                       </div>
                 </motion.div>
            )}
        </AnimatePresence>
      </div> {/* End Results Section */}

       {/* Modals */}
       <ModelInfoModal isOpen={isModelInfoModalOpen} onClose={closeModelInfoModal} model={currentModelInfo} />
       <PresentationDetailModal isOpen={isDetailModalOpen} onClose={closeDetailModal} presentation={currentPresentationDetail} />

    </motion.div>
  );
}

export default DemoPage;