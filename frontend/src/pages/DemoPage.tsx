// frontend/src/pages/DemoPage.tsx
import React, { useState, useCallback, useRef } from 'react';
import UserSelector from '../components/UserSelector';
import RecommendationList from '../components/RecommendationList';
import ErrorMessage from '../components/ErrorMessage';
import SkeletonCard from '../components/SkeletonCard';
import { fetchRecommendations, fetchRandomUser } from '../services/recommendationService';
import { RecommendationItem } from '../types';
import { motion, AnimatePresence } from 'framer-motion';
import { FiRefreshCw, FiHelpCircle, FiUserPlus, FiArrowDownCircle, FiCheckCircle, FiSearch, FiInfo } from 'react-icons/fi';

const DemoPage: React.FC = () => {
  const [selectedUserId, setSelectedUserId] = useState<number | null>(null);
  const [recommendations, setRecommendations] = useState<RecommendationItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isFetchingRandom, setIsFetchingRandom] = useState<boolean>(false);
  const userSelectorRef = useRef<any>(null);
  const resultsRef = useRef<HTMLDivElement>(null); // Ref for the results area

  // --- Fetch Recommendations Logic (with Scroll) ---
  const handleUserSelect = useCallback(async (userId: number | null, scrollIntoView: boolean = false) => {
    if (userId === null) {
        setSelectedUserId(null);
        setRecommendations([]);
        setError(null);
        setIsLoading(false);
        return;
    }
    const numericUserId = parseInt(String(userId), 10);
    if (isNaN(numericUserId)) {
        setError("Invalid User ID selected.");
        return;
    }
    console.log("Fetching recommendations for user:", numericUserId);
    setSelectedUserId(numericUserId);
    setIsLoading(true); // Start loading
    setError(null);
    setRecommendations([]); // Clear previous

    // Scroll slightly below the controls immediately after selection (optional)
    // if (scrollIntoView && resultsRef.current) {
    //    resultsRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    // }

    try {
      const fetchedRecommendations = await fetchRecommendations(numericUserId, 9);
      setRecommendations(fetchedRecommendations);
      setError(null);

       // Scroll after results are fetched and state is likely updated
       if (scrollIntoView && resultsRef.current) {
          setTimeout(() => { // Timeout ensures state update has rendered
             resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start', inline: 'nearest' });
          }, 100); // Adjust delay if needed
       }

    } catch (err: unknown) {
        console.error("API call failed:", err);
        let message = "An unknown error occurred while fetching recommendations.";
        if (err instanceof Error) {
             message = `Failed to fetch recommendations: ${err.message}. Please check the API server and try again.`;
        }
        setError(message);
        setRecommendations([]);
    } finally {
      setIsLoading(false); // Stop loading
    }
  }, []); // Empty dependency array is correct

  // --- Handle Random User Click (Adds scrollIntoView flag) ---
  const handleRandomUser = useCallback(async () => {
    setIsFetchingRandom(true);
    setError(null);
    setRecommendations([]);
    setSelectedUserId(null);
     if (userSelectorRef.current?.select) {
        userSelectorRef.current.select.setValue(null, 'clear');
     }
    try {
        console.log("Fetching random user from API...");
        const randomUser = await fetchRandomUser();
        if (randomUser) {
            const randomId = randomUser.student_id;
            console.log(`Received random user ID: ${randomId}`);
            if (userSelectorRef.current?.select) {
               userSelectorRef.current.select.setValue({ value: randomId, label: String(randomId) }, 'select-option');
             } else { console.warn("Could not set UserSelector value programmatically via ref."); }
             // Call handleUserSelect with the scroll flag set to true
            setTimeout(() => handleUserSelect(randomId, true), 50);
        } else { throw new Error("API did not return a random user."); }
    } catch (err: unknown) {
        console.error("Failed to select random user:", err);
        let message = "Could not select a random user.";
        if (err instanceof Error) { message = `Error selecting random user: ${err.message}`; }
        setError(message);
        setSelectedUserId(null);
        setRecommendations([]);
         if (userSelectorRef.current?.select) { userSelectorRef.current.select.clearValue(); }
    } finally {
        setIsFetchingRandom(false);
    }
  }, [handleUserSelect]); // Include handleUserSelect

  // --- Animation Variants (Unchanged) ---
   const containerVariant = { hidden: { opacity: 0 }, visible: { opacity: 1, transition: { staggerChildren: 0.15, delayChildren: 0.1 } }, };
   const itemVariant = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut'} }, };
   const resultsVariant = { hidden: { opacity: 0, scale: 0.97 }, visible: { opacity: 1, scale: 1, transition: { duration: 0.4, ease: "easeOut" } }, exit: { opacity: 0, scale: 0.97, transition: { duration: 0.2, ease: "easeIn" } } };


  return (
    <motion.div
        className="container mx-auto px-4 py-16 md:py-20 max-w-6xl" // Slightly wider max-width for content
        variants={containerVariant} initial="hidden" animate="visible"
    >
      {/* Header */}
      <motion.header className="text-center mb-12 md:mb-16" variants={itemVariant}>
        <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold text-text-primary mb-4 tracking-tight">
          Interactive Recommendation Demo
        </h1>
        <p className="text-text-secondary md:text-lg max-w-3xl mx-auto">
           Choose a student ID or pick one randomly to view course suggestions from our Item-Based Collaborative Filtering model.
        </p>
      </motion.header>

      {/* Control Section */}
      <motion.div className="max-w-lg mx-auto mb-16 md:mb-20" variants={itemVariant}> {/* Narrower controls */}
          <div className="bg-surface rounded-xl shadow-lg border border-border-color p-6">
            {/* FIX: Added items-center for alignment */}
            <div className="flex items-center gap-3 mb-4">
                <FiSearch className="text-primary text-xl flex-shrink-0" />
                 <h2 className="text-lg font-semibold text-text-primary">Select a Student</h2>
            </div>
             <UserSelector
               ref={userSelectorRef}
               // Pass handleUserSelect directly, scroll flag is false by default
               onUserSelect={(userId) => handleUserSelect(userId, false)}
               isLoading={isLoading || isFetchingRandom}
             />
             <div className="flex items-center my-4"> {/* OR Divider */}
                 <span className="flex-grow border-t border-border-color opacity-50"></span>
                 <span className="flex-shrink mx-3 text-xs text-text-muted">OR</span>
                 <span className="flex-grow border-t border-border-color opacity-50"></span>
             </div>
             <div className="text-center">
                  <motion.button
                      onClick={handleRandomUser}
                      disabled={isFetchingRandom || isLoading}
                      className="btn btn-secondary btn-sm w-full disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                      whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.98 }}
                  >
                      <FiRefreshCw size={16} className={`${isFetchingRandom ? 'animate-spin' : ''}`} />
                      {isFetchingRandom ? 'Selecting...' : 'Try a Random Student ID'}
                  </motion.button>
             </div>
           </div>
      </motion.div>

      {/* --- Results Section --- */}
      <motion.div
        ref={resultsRef} // Attach ref here
        className="min-h-[500px] relative bg-surface rounded-xl shadow-xl border border-border-color p-6 md:p-10" // Added background and padding to results area
        variants={itemVariant}
       >
        <AnimatePresence mode="wait">
          {/* Initial State: Shown only if no user selected AND not loading/error */}
          {!isLoading && !error && !selectedUserId && (
             <motion.div
                key="initial-prompt"
                variants={resultsVariant} initial="hidden" animate="visible" exit="exit"
                className="flex flex-col items-center justify-center text-center text-text-muted h-full pt-10 pb-10 min-h-[300px]" // Ensure height
              >
                 <FiArrowDownCircle size={48} className="mx-auto mb-5 text-border-color opacity-60 animate-pulse" />
                 <p className="text-lg font-medium text-text-secondary">Waiting for Input</p>
                 <p className="text-sm mt-1">Select a student above to see recommendations.</p>
             </motion.div>
          )}

          {/* Loading State */}
          {isLoading && selectedUserId && (
            <motion.div key="loading" variants={resultsVariant} initial="hidden" animate="visible" exit="exit" className="w-full text-center">
                <h2 className="text-xl font-semibold mb-8 text-center text-text-muted animate-pulse">
                    Generating Recommendations for Student <span className='font-bold text-primary'>{selectedUserId}</span>...
                </h2>
               <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8">
                 {Array.from({ length: 6 }).map((_, index) => ( <SkeletonCard key={`skel-${index}`} /> ))}
               </div>
            </motion.div>
          )}

          {/* Error State */}
          {!isLoading && error && (
            <motion.div key="error" variants={resultsVariant} initial="hidden" animate="visible" exit="exit">
              <ErrorMessage message={error} />
            </motion.div>
          )}

          {/* Results State: Display only if we have a user AND not loading AND no error */}
          {!isLoading && !error && selectedUserId && (
            <motion.div key={`results-${selectedUserId}`} variants={resultsVariant} initial="hidden" animate="visible" exit="exit">
              {/* --- REINTRODUCED Context Title/Explanation --- */}
               <div className="text-center mb-10">
                 <h2 className="text-2xl font-semibold mb-3 text-text-primary">
                   Recommendations for Student <span className='font-bold text-primary'>{selectedUserId}</span>
                 </h2>
                 <p className="text-sm text-text-muted max-w-xl mx-auto">
                   Based on collaborative filtering, these courses are similar to ones previously engaged with by this student and their peers.
                 </p>
               </div>
               {/* Recommendation List */}
               <RecommendationList
                 recommendations={recommendations}
                 selectedUserId={selectedUserId} // Still needed for empty state logic inside the list
               />
                {recommendations.length > 0 && (
                  <p className="text-xs text-text-muted text-center mt-10 italic flex items-center justify-center gap-1.5">
                    <FiInfo size={13}/> Hover over the score on a card for the model's confidence value.
                  </p>
                 )}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </motion.div>
  );
}

export default DemoPage;