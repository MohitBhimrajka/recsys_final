// frontend/src/pages/DemoPage.tsx
import React, { useState, useCallback, useRef, useReducer } from 'react';
import UserSelector from '../components/UserSelector';
import RecommendationList from '../components/RecommendationList';
import ErrorMessage from '../components/ErrorMessage';
import SkeletonCard from '../components/SkeletonCard';
import ModelInfoModal from '../components/ModelInfoModal';
import PresentationDetailModal from '../components/PresentationDetailModal';
import { fetchRecommendations, fetchAllModelRecommendations, fetchRandomUser } from '../services/recommendationService';
import { RecommendationItem, AllModelsRecs, modelInfos, findModelInfoByName, ModelInfo, PresentationDetailInfo } from '../types';
import { motion, AnimatePresence } from 'framer-motion';
import { FiRefreshCw, FiInfo, FiSearch, FiArrowDownCircle, FiLayers, FiBarChart2, FiGrid, FiCpu, FiSliders, FiInbox, FiChevronRight } from 'react-icons/fi'; // Adjusted icons
// Import the new Analysis Dashboard component
import AnalysisDashboard from '../components/AnalysisDashboard';

// --- TYPES & CONSTANTS ---
type TabId = 'ensemble' | 'itemcf' | 'ncf' | 'hybrid' | 'als' | 'popularity' | 'analysis';

// Updated TABS_CONFIG to match modelInfos and desired order
const TABS_CONFIG: { id: TabId; label: string; icon: React.ReactNode; order: number, modelInfoName?: string }[] = [
    { id: 'ensemble' as TabId, label: 'Combined Suggestion', icon: <FiLayers size={16}/>, order: 1 },
    { id: 'analysis' as TabId, label: 'Analysis & Comparison', icon: <FiBarChart2 size={16}/>, order: 2 },
    { id: 'itemcf' as TabId, label: 'ItemCF', icon: <FiGrid size={16}/>, order: 3, modelInfoName: 'ItemCF' },
    { id: 'ncf' as TabId, label: 'NCF', icon: <FiCpu size={16}/>, order: 4, modelInfoName: 'NCF (e=15)' },
    { id: 'hybrid' as TabId, label: 'Hybrid', icon: <FiSliders size={16}/>, order: 5, modelInfoName: 'Hybrid (e=15)' },
    { id: 'als' as TabId, label: 'ALS', icon: <FiSliders size={16}/>, order: 6, modelInfoName: 'ALS (f=100)' },
    { id: 'popularity' as TabId, label: 'Popularity', icon: <FiInbox size={16}/>, order: 7, modelInfoName: 'Popularity' },
].sort((a, b) => a.order - b.order);


// Card State Reducer (Unchanged)
type CardStateAction = | { type: 'HIDE'; payload: string } | { type: 'TOGGLE_HIGHLIGHT'; payload: string } | { type: 'RESET' };
interface CardDisplayState { hiddenCards: Set<string>; highlightedCards: Set<string>; }
function cardDisplayReducer(state: CardDisplayState, action: CardStateAction): CardDisplayState {
    switch (action.type) {
        case 'HIDE': return { ...state, hiddenCards: new Set(state.hiddenCards).add(action.payload) };
        case 'TOGGLE_HIGHLIGHT': { const newHighlighted = new Set(state.highlightedCards); if (newHighlighted.has(action.payload)) newHighlighted.delete(action.payload); else newHighlighted.add(action.payload); return { ...state, highlightedCards: newHighlighted }; }
        case 'RESET': return { hiddenCards: new Set(), highlightedCards: new Set() };
        default: return state;
    }
}

// --- Tab Component (No Changes) ---
interface TabProps { label: string; icon?: React.ReactNode; isActive: boolean; onClick: () => void; highlight?: boolean; }
const Tab: React.FC<TabProps> = ({ label, icon, isActive, onClick, highlight = false }) => {
    const baseClasses = "relative flex-shrink-0 flex items-center gap-2 px-4 sm:px-5 py-3 text-sm font-medium transition-colors duration-200 outline-none focus-visible:ring-1 focus-visible:ring-primary rounded-t-md";
    const activeClasses = "text-primary";
    const inactiveBaseClasses = "text-text-muted hover:text-text-secondary";
    const inactiveHighlightClasses = "bg-surface/50 hover:bg-surface/80";
    const inactiveNormalClasses = "hover:bg-transparent";
    const inactiveCombinedClasses = `${inactiveBaseClasses} ${highlight && !isActive ? inactiveHighlightClasses : inactiveNormalClasses}`;
    return (
        <motion.button onClick={onClick} role="tab" aria-selected={isActive} className={`${baseClasses} ${isActive ? activeClasses : inactiveCombinedClasses}`} whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.98 }}>
            {icon} <span className="whitespace-nowrap">{label}</span>
            {isActive && ( <motion.div className="absolute bottom-[-1px] left-0 right-0 h-0.5 bg-primary" layoutId="activeTabIndicator" initial={false} transition={{ type: 'spring', stiffness: 300, damping: 25 }} /> )}
        </motion.button>
    );
};

// --- Helper for Score Distribution Info (No Changes) ---
const ScoreDistributionInfo: React.FC<{ scores: number[] }> = ({ scores }) => {
    if (!scores || scores.length === 0) return null;
    const minScore = Math.min(...scores).toFixed(4);
    const maxScore = Math.max(...scores).toFixed(4);
    const avgScore = (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(4);
    return ( <div className="mt-6 text-center text-xs text-text-muted border-t border-border-color/50 pt-4"> Top {scores.length} Scores - Min: <strong className="text-text-secondary">{minScore}</strong> | Avg: <strong className="text-text-secondary">{avgScore}</strong> | Max: <strong className="text-text-secondary">{maxScore}</strong> </div> );
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

  // --- Refs ---
  const userSelectorRef = useRef<any>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  // --- Modal Handlers --- (No Changes)
  const openModelInfoModal = (modelName: string) => { const modelInfo = findModelInfoByName(modelName); if (modelInfo) { setCurrentModelInfo(modelInfo); setIsModelInfoModalOpen(true); }};
  const closeModelInfoModal = () => setIsModelInfoModalOpen(false);
  const openDetailModal = (presentation: PresentationDetailInfo) => { setCurrentPresentationDetail(presentation); setIsDetailModalOpen(true); };
  const closeDetailModal = () => setIsDetailModalOpen(false);

  // --- Data Fetching Logic --- (No Changes)
  const handleUserSelect = useCallback(async (userId: number | null, scrollIntoView: boolean = false) => {
    dispatchCardState({ type: 'RESET' });
    setEnsembleRecommendations(null); setAllModelRecommendations(null); setActiveTab('ensemble');
    // setSelectedComparisonItem(null); // Reset logic moved to AnalysisDashboard

    if (userId === null) { setSelectedUserId(null); setError(null); setIsLoading(false); return; }
    const numericUserId = parseInt(String(userId), 10);
    if (isNaN(numericUserId)) { setError("Invalid User ID selected."); setIsLoading(false); return; }
    console.log("Fetching recommendations for user:", numericUserId);
    setSelectedUserId(numericUserId); setIsLoading(true); setError(null);
    try {
      // Fetch Top 9 for display
      const [ensembleResult, allModelsResult] = await Promise.allSettled([ fetchRecommendations(numericUserId, 9), fetchAllModelRecommendations(numericUserId, 9) ]);
      let fetchError: string | null = null;
      if (ensembleResult.status === 'fulfilled') setEnsembleRecommendations(ensembleResult.value); else { console.error("Failed ensemble:", ensembleResult.reason); fetchError = `Failed combined suggestions: ${ensembleResult.reason?.message || 'Unknown error'}. `; }
      if (allModelsResult.status === 'fulfilled') setAllModelRecommendations(allModelsResult.value); else { console.error("Failed all models:", allModelsResult.reason); fetchError = (fetchError || "") + `Failed individual model results: ${allModelsResult.reason?.message || 'Unknown error'}.`; }
      if (fetchError) setError(fetchError + " Please check the API server."); else { setError(null); if (scrollIntoView && resultsRef.current) setTimeout(() => { resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }); }, 150); }
    } catch (err: unknown) { console.error("Fetch error:", err); setError("An unexpected error occurred while fetching data."); setEnsembleRecommendations(null); setAllModelRecommendations(null); }
    finally { setIsLoading(false); }
   }, []);

  const handleRandomUser = useCallback(async () => {
    dispatchCardState({ type: 'RESET' }); setIsFetchingRandom(true); setError(null);
    setEnsembleRecommendations(null); setAllModelRecommendations(null); setSelectedUserId(null);
    // setSelectedComparisonItem(null); // Reset logic moved to AnalysisDashboard
    if (userSelectorRef.current?.select) userSelectorRef.current.select.setValue(null, 'clear');
    try { const randomUser = await fetchRandomUser(); if (randomUser) { const randomId = randomUser.student_id; if (userSelectorRef.current?.select) userSelectorRef.current.select.setValue({ value: randomId, label: String(randomId) }, 'select-option'); setTimeout(() => handleUserSelect(randomId, true), 50); } else throw new Error("API did not return a random user."); }
    catch (err: unknown) { console.error("Random user fetch failed:", err); setError(`Error selecting random user: ${err instanceof Error ? err.message : 'Unknown error'}.`); setSelectedUserId(null); if (userSelectorRef.current?.select) userSelectorRef.current.select.clearValue(); }
    finally { setIsFetchingRandom(false); }
  }, [handleUserSelect]);

  // --- Card Interaction Handlers --- (No Changes)
  const handleHideCard = (presentationId: string) => dispatchCardState({ type: 'HIDE', payload: presentationId });
  const handleHighlightCard = (presentationId: string) => dispatchCardState({ type: 'TOGGLE_HIGHLIGHT', payload: presentationId });
  const handleCardClick = (recommendation: RecommendationItem) => openDetailModal(recommendation);

  // --- Animation Variants --- (No Changes)
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
                            <li><FiChevronRight className="inline mr-1 opacity-70"/> The <strong className="text-text-secondary">Analysis & Comparison</strong> tab provides tools to directly compare model outputs side-by-side.</li>
                            <li><FiChevronRight className="inline mr-1 opacity-70"/> The <strong className="text-text-secondary">Individual Model</strong> tabs (ItemCF, NCF, etc.) show raw rankings from each algorithm.</li>
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
                                                // Use the modelInfoName from TABS_CONFIG to get the correct key for allModelRecommendations
                                                const modelKey = tab.modelInfoName || tab.label; // Fallback to label if name not specified
                                                const modelInfo = findModelInfoByName(modelKey);
                                                const modelRecs = allModelRecommendations ? allModelRecommendations[modelKey] : [];
                                                return (
                                                    <div className="space-y-8 md:space-y-10">
                                                         <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-1">
                                                             <button onClick={() => modelInfo && openModelInfoModal(modelInfo.name)} disabled={!modelInfo} className="text-xl md:text-2xl font-semibold text-text-primary flex items-center gap-2 hover:text-primary disabled:hover:text-text-primary disabled:opacity-60 transition-colors outline-none focus-visible:ring-1 focus-visible:ring-primary rounded text-left" title={modelInfo ? `Learn about the ${modelInfo.name} model` : tab.label}> {tab.label} {modelInfo && <FiInfo size={18} className="opacity-70"/>} </button>
                                                             {modelInfo?.evaluationScore && ( <p className="text-xs text-left sm:text-right text-text-muted italic whitespace-nowrap">Offline NDCG@10: <span className="font-medium text-text-secondary">{modelInfo.evaluationScore.toFixed(4)}</span></p> )}
                                                        </div>
                                                        {modelInfo?.description && ( <p className="text-sm text-text-muted mb-6">{modelInfo.description.split('.')[0]}.</p> )}
                                                         <RecommendationList recommendations={modelRecs || []} selectedUserId={selectedUserId} onCardClick={handleCardClick} isIndividualModelList={true} modelName={modelKey} />
                                                     </div> );
                                           })()}

                                           {/* --- Analysis Tab --- */}
                                           {tab.id === 'analysis' && (
                                                <AnalysisDashboard
                                                    allRecs={allModelRecommendations}
                                                    ensembleRecs={ensembleRecommendations}
                                                    selectedUserId={selectedUserId}
                                                    modelColors={modelInfos.reduce((acc, m) => ({ ...acc, [m.name]: m.id === 'itemcf' ? 'var(--color-secondary-default)' : (m.id === 'ncf' ? '#10b981' : (m.id === 'hybrid' ? '#f59e0b' : (m.id === 'als' ? '#ef4444' : (m.id === 'popularity' ? '#6b7280' : 'var(--color-primary-default)')))) }), { 'Combined': 'var(--color-primary-default)' })}
                                                />
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