// frontend/src/pages/DemoPage.tsx
import React, { useState, useCallback, useRef, useReducer } from 'react';
import UserSelector from '../components/UserSelector';
import RecommendationList from '../components/RecommendationList';
import ErrorMessage from '../components/ErrorMessage';
import SkeletonCard from '../components/SkeletonCard';
import ModelInfoModal from '../components/ModelInfoModal'; // Import modal
import { fetchRecommendations, fetchAllModelRecommendations, fetchRandomUser } from '../services/recommendationService';
import { RecommendationItem, AllModelsRecs, modelInfos, findModelInfoByName, ModelInfo } from '../types'; // Import types
import { motion, AnimatePresence } from 'framer-motion';
import { FiRefreshCw, FiHelpCircle, FiUserPlus, FiArrowDownCircle, FiCheckCircle, FiSearch, FiInfo, FiGrid, FiLayers } from 'react-icons/fi'; // Added icons

// Reducer for managing hidden/highlighted card states (applied only to ensemble list for simplicity)
type CardStateAction =
  | { type: 'HIDE'; payload: string }
  | { type: 'TOGGLE_HIGHLIGHT'; payload: string }
  | { type: 'RESET' };

interface CardDisplayState {
    hiddenCards: Set<string>;
    highlightedCards: Set<string>;
}

function cardDisplayReducer(state: CardDisplayState, action: CardStateAction): CardDisplayState {
    switch (action.type) {
        case 'HIDE': return { ...state, hiddenCards: new Set(state.hiddenCards).add(action.payload) };
        case 'TOGGLE_HIGHLIGHT': {
            const newHighlighted = new Set(state.highlightedCards);
            if (newHighlighted.has(action.payload)) newHighlighted.delete(action.payload);
            else newHighlighted.add(action.payload);
            return { ...state, highlightedCards: newHighlighted };
        }
        case 'RESET': return { hiddenCards: new Set(), highlightedCards: new Set() };
        default: return state;
    }
}

const DemoPage: React.FC = () => {
  const [selectedUserId, setSelectedUserId] = useState<number | null>(null);
  const [ensembleRecommendations, setEnsembleRecommendations] = useState<RecommendationItem[] | null>(null);
  const [allModelRecommendations, setAllModelRecommendations] = useState<AllModelsRecs | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false); // Unified loading state
  const [error, setError] = useState<string | null>(null);
  const [isFetchingRandom, setIsFetchingRandom] = useState<boolean>(false);

  const [cardState, dispatchCardState] = useReducer(cardDisplayReducer, {
      hiddenCards: new Set<string>(),
      highlightedCards: new Set<string>(),
  });

  // Modal State
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [currentModalModel, setCurrentModalModel] = useState<ModelInfo | null>(null);

  const userSelectorRef = useRef<any>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  const openModelInfoModal = (modelName: string) => {
      const modelInfo = findModelInfoByName(modelName);
      if (modelInfo) {
          setCurrentModalModel(modelInfo);
          setIsModalOpen(true);
      } else {
          console.warn(`ModelInfo not found for name: ${modelName}`);
      }
  };
  const closeModelInfoModal = () => setIsModalOpen(false);


  // --- Fetch Recommendations Logic (Fetches BOTH Ensemble and All Models) ---
  const handleUserSelect = useCallback(async (userId: number | null, scrollIntoView: boolean = false) => {
    dispatchCardState({ type: 'RESET' }); // Reset card states
    setEnsembleRecommendations(null); // Clear previous results
    setAllModelRecommendations(null);

    if (userId === null) {
        setSelectedUserId(null);
        setError(null);
        setIsLoading(false);
        return;
    }

    const numericUserId = parseInt(String(userId), 10);
    if (isNaN(numericUserId)) {
        setError("Invalid User ID selected.");
        setIsLoading(false);
        return;
    }

    console.log("Fetching recommendations for user:", numericUserId);
    setSelectedUserId(numericUserId);
    setIsLoading(true);
    setError(null);

    try {
      // Fetch both results concurrently
      const [ensembleResult, allModelsResult] = await Promise.allSettled([
          fetchRecommendations(numericUserId, 9), // Ensemble
          fetchAllModelRecommendations(numericUserId, 9) // All models
      ]);

      let fetchError: string | null = null;

      if (ensembleResult.status === 'fulfilled') {
          setEnsembleRecommendations(ensembleResult.value);
      } else {
          console.error("Failed to fetch ensemble recommendations:", ensembleResult.reason);
          fetchError = `Failed to fetch main recommendations: ${ensembleResult.reason?.message || 'Unknown error'}. `;
      }

      if (allModelsResult.status === 'fulfilled') {
          setAllModelRecommendations(allModelsResult.value);
      } else {
          console.error("Failed to fetch all model recommendations:", allModelsResult.reason);
          fetchError = (fetchError || "") + `Failed to fetch individual model recommendations: ${allModelsResult.reason?.message || 'Unknown error'}.`;
      }

       if (fetchError) {
           setError(fetchError + " Please check the API server.");
       } else {
           setError(null); // Clear previous errors if both succeed
           // Scroll only if fetches were successful and requested
           if (scrollIntoView && resultsRef.current) {
               setTimeout(() => {
                   resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
               }, 100);
           }
       }

    } catch (err: unknown) { // Catch potential errors in Promise.allSettled itself (unlikely)
        console.error("Unexpected error during fetch:", err);
        setError("An unexpected error occurred. Please try again.");
        setEnsembleRecommendations(null);
        setAllModelRecommendations(null);
    } finally {
      setIsLoading(false);
    }
  }, []); // handleUserSelect dependencies are implicitly managed by useCallback

  // --- Handle Random User Click (Updated) ---
  const handleRandomUser = useCallback(async () => {
    dispatchCardState({ type: 'RESET' });
    setIsFetchingRandom(true);
    setError(null);
    setEnsembleRecommendations(null);
    setAllModelRecommendations(null);
    setSelectedUserId(null);
    if (userSelectorRef.current?.select) {
        userSelectorRef.current.select.setValue(null, 'clear');
    }
    try {
        const randomUser = await fetchRandomUser();
        if (randomUser) {
            const randomId = randomUser.student_id;
            if (userSelectorRef.current?.select) {
               userSelectorRef.current.select.setValue({ value: randomId, label: String(randomId) }, 'select-option');
             }
            // Trigger the combined fetch after setting the value
            setTimeout(() => handleUserSelect(randomId, true), 50);
        } else { throw new Error("API did not return a random user."); }
    } catch (err: unknown) {
        console.error("Failed to select random user:", err);
        setError(`Error selecting random user: ${err instanceof Error ? err.message : 'Unknown error'}.`);
        setSelectedUserId(null);
        if (userSelectorRef.current?.select) { userSelectorRef.current.select.clearValue(); }
    } finally {
        setIsFetchingRandom(false);
    }
  }, [handleUserSelect]); // Depends on handleUserSelect

  // --- Card Interaction Handlers (Applied to Ensemble List) ---
  const handleHideCard = (presentationId: string) => dispatchCardState({ type: 'HIDE', payload: presentationId });
  const handleHighlightCard = (presentationId: string) => dispatchCardState({ type: 'TOGGLE_HIGHLIGHT', payload: presentationId });
  const handleCardClick = (recommendation: RecommendationItem) => {
      console.log("Ensemble Card clicked:", recommendation);
      alert(`Ensemble Rec Clicked: ${recommendation.presentation_id}\nScore: ${recommendation.score.toFixed(3)}`);
  };
  const handleIndividualCardClick = (modelName: string, recommendation: RecommendationItem) => {
      console.log(`Individual Card clicked (${modelName}):`, recommendation);
       alert(`${modelName} Rec Clicked: ${recommendation.presentation_id}\nScore: ${recommendation.score.toFixed(3)}`);
  };

  // --- Animation Variants ---
   const containerVariant = { hidden: { opacity: 0 }, visible: { opacity: 1, transition: { staggerChildren: 0.15, delayChildren: 0.1 } }, };
   const itemVariant = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut'} }, };
   const resultsVariant = { hidden: { opacity: 0 }, visible: { opacity: 1, transition: { duration: 0.4, ease: "easeOut" } }, exit: { opacity: 0, transition: { duration: 0.2, ease: "easeIn" } } };

  // Check if individual model results are available and non-empty
   const hasIndividualResults = allModelRecommendations && Object.keys(allModelRecommendations).length > 0 && Object.values(allModelRecommendations).some(recs => recs.length > 0);

  return (
    <motion.div
        className="container mx-auto px-4 py-16 md:py-20 max-w-7xl" // Wider max width
        variants={containerVariant} initial="hidden" animate="visible"
    >
      {/* Header */}
      <motion.header className="text-center mb-12 md:mb-16" variants={itemVariant}>
        <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold text-text-primary mb-4 tracking-tight">
          Multi-Model Recommendation Demo
        </h1>
        <p className="text-text-secondary md:text-lg max-w-3xl mx-auto">
           Select a student to view combined course suggestions (weighted ensemble) and compare results from individual models like ItemCF, ALS, NCF, and Hybrid.
        </p>
      </motion.header>

      {/* Control Section */}
      <motion.div className="max-w-lg mx-auto mb-16 md:mb-20" variants={itemVariant}>
          <div className="bg-surface rounded-xl shadow-lg border border-border-color p-6">
             <div className="flex items-center gap-3 mb-4">
                <FiSearch className="text-primary text-xl flex-shrink-0" />
                 <h2 className="text-lg font-semibold text-text-primary">Select a Student</h2>
            </div>
             <UserSelector ref={userSelectorRef} onUserSelect={(userId) => handleUserSelect(userId, true)} isLoading={isLoading || isFetchingRandom} />
             <div className="flex items-center my-4">
                 <span className="flex-grow border-t border-border-color opacity-50"></span>
                 <span className="flex-shrink mx-3 text-xs text-text-muted">OR</span>
                 <span className="flex-grow border-t border-border-color opacity-50"></span>
             </div>
             <motion.button onClick={handleRandomUser} disabled={isFetchingRandom || isLoading} className="btn btn-secondary btn-sm w-full disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2" whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.98 }} >
                 <FiRefreshCw size={16} className={`${isFetchingRandom ? 'animate-spin' : ''}`} />
                 {isFetchingRandom ? 'Selecting...' : 'Try a Random Student ID'}
             </motion.button>
           </div>
      </motion.div>

      {/* Results Section Wrapper */}
      <div ref={resultsRef} className="space-y-16 md:space-y-20">
        <AnimatePresence mode="wait">
            {/* --- Initial State --- */}
            {!isLoading && !error && !selectedUserId && (
                 <motion.div key="initial-prompt" variants={resultsVariant} initial="hidden" animate="visible" exit="exit" className="text-center text-text-muted pt-10 pb-10 min-h-[300px]">
                     <motion.div initial={{ y: -5 }} animate={{ y: [0, -5, 0], transition: { duration: 1.5, repeat: Infinity, ease: "easeInOut" } }}>
                          <FiArrowDownCircle size={48} className="mx-auto mb-5 text-border-color opacity-60" />
                     </motion.div>
                     <p className="text-lg font-medium text-text-secondary">Waiting for Input</p>
                     <p className="text-sm mt-1">Select a student above to see recommendations.</p>
                 </motion.div>
            )}

            {/* --- Loading State --- */}
            {isLoading && selectedUserId && (
                 <motion.div key="loading" variants={resultsVariant} initial="hidden" animate="visible" exit="exit" className="w-full">
                     <h2 className="text-xl font-semibold mb-8 text-center text-text-muted animate-pulse">
                        Generating Recommendations for Student <span className='font-bold text-primary'>{selectedUserId}</span>...
                     </h2>
                     {/* Show skeletons for both sections potentially */}
                     <div className="space-y-12">
                          <div>
                              <h3 className="text-lg font-medium text-text-secondary text-center mb-4">Loading Combined Results...</h3>
                              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8">
                                  {Array.from({ length: 3 }).map((_, index) => <SkeletonCard key={`skel-ens-${index}`} />)}
                              </div>
                          </div>
                          <div>
                               <h3 className="text-lg font-medium text-text-secondary text-center mb-4">Loading Individual Model Results...</h3>
                               <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8">
                                   {Array.from({ length: 3 }).map((_, index) => <SkeletonCard key={`skel-ind-${index}`} />)}
                               </div>
                          </div>
                     </div>
                 </motion.div>
            )}

             {/* --- Error State --- */}
             {!isLoading && error && (
                 <motion.div key="error" variants={resultsVariant} initial="hidden" animate="visible" exit="exit">
                     <ErrorMessage message={error} />
                 </motion.div>
             )}

              {/* --- Results Display State --- */}
             {!isLoading && !error && selectedUserId && (
                 <motion.div key={`results-${selectedUserId}`} variants={resultsVariant} initial="hidden" animate="visible" exit="exit" className="space-y-16 md:space-y-20">

                     {/* 1. Ensemble Results Section */}
                     <section className="bg-surface rounded-xl shadow-xl border border-border-color p-6 md:p-10">
                         <div className="text-center mb-10">
                             <h2 className="text-2xl font-semibold mb-3 text-text-primary flex items-center justify-center gap-2">
                                 <FiLayers className="text-primary"/> Combined Recommendations (Ensemble)
                             </h2>
                             <p className="text-sm text-text-muted max-w-xl mx-auto">
                                 Top suggestions based on a weighted average of scores from all models, prioritizing ItemCF and NCF.
                             </p>
                         </div>
                         {ensembleRecommendations !== null ? (
                             <RecommendationList
                                 recommendations={ensembleRecommendations}
                                 selectedUserId={selectedUserId}
                                 hiddenCards={cardState.hiddenCards}
                                 highlightedCards={cardState.highlightedCards}
                                 onHideCard={handleHideCard}
                                 onHighlightCard={handleHighlightCard}
                                 onCardClick={handleCardClick}
                             />
                         ) : (
                              <p className="text-center text-text-muted">No combined recommendations could be generated.</p>
                         )}
                          {ensembleRecommendations && ensembleRecommendations.length > 0 && (
                             <p className="text-xs text-text-muted text-center mt-10 italic flex items-center justify-center gap-1.5">
                                 <FiInfo size={13}/> Interact with cards above (hide/highlight/click).
                             </p>
                         )}
                     </section>

                     {/* 2. Individual Model Results Section */}
                     {hasIndividualResults && (
                         <section>
                              <div className="text-center mb-10">
                                 <h2 className="text-2xl font-semibold mb-3 text-text-primary flex items-center justify-center gap-2">
                                      <FiGrid className="text-primary"/> Individual Model Results
                                 </h2>
                                 <p className="text-sm text-text-muted max-w-xl mx-auto">
                                     Compare the top suggestions from each underlying model. Click model name for details.
                                 </p>
                             </div>
                             <div className="space-y-12">
                                 {allModelRecommendations && Object.entries(allModelRecommendations)
                                     // Optional: Sort models for consistent display order
                                     .sort(([nameA], [nameB]) => modelInfos.findIndex(m => m.name === nameA) - modelInfos.findIndex(m => m.name === nameB))
                                     .map(([modelName, recs]) => (
                                         <div key={modelName} className="bg-surface/50 rounded-lg border border-border-color/50 p-5 md:p-6">
                                             <h3 className="text-xl font-semibold text-text-primary mb-5 flex items-center gap-2">
                                                 {modelName}
                                                 <button onClick={() => openModelInfoModal(modelName)} className="text-text-muted hover:text-primary transition-colors" title={`About ${modelName}`}>
                                                     <FiInfo size={16}/>
                                                 </button>
                                             </h3>
                                             {recs.length > 0 ? (
                                                 <RecommendationList
                                                     recommendations={recs}
                                                     selectedUserId={selectedUserId}
                                                     // Interactions disabled for individual lists for simplicity
                                                     hiddenCards={new Set()}
                                                     highlightedCards={new Set()}
                                                     onHideCard={() => {}}
                                                     onHighlightCard={() => {}}
                                                     // Use different click handler for individual cards
                                                     onCardClick={(rec) => handleIndividualCardClick(modelName, rec)}
                                                 />
                                             ) : (
                                                  <p className="text-sm text-center text-text-muted italic py-4">No recommendations generated by this model for this user.</p>
                                             )}
                                         </div>
                                 ))}
                             </div>
                         </section>
                     )}
                 </motion.div>
            )}
        </AnimatePresence>
      </div> {/* End Results Section Wrapper */}

       {/* Modal Component */}
       <ModelInfoModal
           isOpen={isModalOpen}
           onClose={closeModelInfoModal}
           model={currentModalModel}
       />

    </motion.div>
  );
}

export default DemoPage;