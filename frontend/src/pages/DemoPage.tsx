// frontend/src/pages/DemoPage.tsx
import React, { useState, useCallback } from 'react';
import UserSelector from '../components/UserSelector';
import RecommendationList from '../components/RecommendationList';
import ErrorMessage from '../components/ErrorMessage';
import SkeletonCard from '../components/SkeletonCard';
import { fetchRecommendations } from '../services/recommendationService';
import { RecommendationItem } from '../types';

const DemoPage: React.FC = () => {
  const [selectedUserId, setSelectedUserId] = useState<number | null>(null);
  const [recommendations, setRecommendations] = useState<RecommendationItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleUserSelect = useCallback(async (userId: number) => {
    console.log("Fetching recommendations for user:", userId);
    setSelectedUserId(userId);
    setIsLoading(true);
    setError(null); // Clear previous errors
    setRecommendations([]); // Clear previous recommendations immediately

    try {
      // Optional: Simulate network delay
      // await new Promise(resolve => setTimeout(resolve, 1000));
      const fetchedRecommendations = await fetchRecommendations(userId, 10); // Fetch top 10
      setRecommendations(fetchedRecommendations);
      // Clear error on successful fetch, even if recommendations array is empty
      setError(null);
    } catch (err: unknown) { // Catch unknown type
        console.error("API call failed:", err);
        // Provide a user-friendly message
        if (err instanceof Error) {
             setError(`Failed to fetch recommendations: ${err.message}. Please check the API server and try again.`);
        } else {
            setError("An unknown error occurred while fetching recommendations.");
        }
        setRecommendations([]); // Ensure recommendations are cleared on error
    } finally {
      setIsLoading(false);
    }
  }, []);

  return (
    // Using container/max-width from Layout now, so no need here unless overriding
    <div>
      <header className="text-center mb-8">
        <h1 className="text-2xl font-semibold text-gray-800">
          Recommendation Demo
        </h1>
        <p className="text-gray-600 mt-2">
          Select a student ID using the searchable dropdown to view personalized course presentation recommendations based on the ItemCF model.
        </p>
      </header>

      {/* Pass the main isLoading state to UserSelector for disabling */}
      <UserSelector onUserSelect={handleUserSelect} isLoading={isLoading} />

      <div className="mt-6 min-h-[300px]"> {/* Added min-height to prevent layout jump */}
        {/* Show skeleton ONLY when loading AND a user has been selected */}
        {isLoading && selectedUserId && (
          <div>
            <h2 className="text-xl font-semibold mb-4 text-center text-gray-500 animate-pulse">
                Loading Recommendations for Student {selectedUserId}...
            </h2>
            {/* Skeleton Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {Array.from({ length: 6 }).map((_, index) => (
                <SkeletonCard key={index} />
              ))}
            </div>
          </div>
        )}

        {/* Show error message if there's an error AND not loading */}
        {!isLoading && error && <ErrorMessage message={error} />}

        {/* Show RecommendationList ONLY when NOT loading, no error, and a user IS selected */}
        {/* The list component itself handles the "no recommendations found" message */}
        {!isLoading && !error && selectedUserId && (
          <RecommendationList
            recommendations={recommendations}
            selectedUserId={selectedUserId} // Pass user ID for context in message
          />
        )}

        {/* Initial state message when no user is selected AND not loading */}
         {!isLoading && !error && !selectedUserId && (
             <p className="text-center text-gray-500 mt-10 text-lg">
                Please select a student ID above to get started.
            </p>
         )}
      </div>
    </div>
  );
}

export default DemoPage;