// frontend/src/App.tsx
import React, { useState, useCallback } from 'react';
import UserSelector from './components/UserSelector';
import RecommendationList from './components/RecommendationList';
import LoadingSpinner from './components/LoadingSpinner';
import ErrorMessage from './components/ErrorMessage';
import { fetchRecommendations } from './services/recommendationService';
import { RecommendationItem } from './types';

function App() {
  const [selectedUserId, setSelectedUserId] = useState<number | null>(null);
  const [recommendations, setRecommendations] = useState<RecommendationItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleUserSelect = useCallback(async (userId: number) => {
    console.log("Fetching recommendations for user:", userId);
    setSelectedUserId(userId);
    setIsLoading(true);
    setError(null);
    setRecommendations([]); // Clear previous recommendations

    try {
      const fetchedRecommendations = await fetchRecommendations(userId, 10); // Fetch top 10
      setRecommendations(fetchedRecommendations);
    } catch (err) {
      console.error("API call failed:", err);
      setError("Failed to fetch recommendations. Please check the API server and try again.");
      setRecommendations([]); // Clear recommendations on error
    } finally {
      setIsLoading(false);
    }
  }, []); // Empty dependency array means this function doesn't change

  return (
    <div className="container mx-auto p-4 pt-8 max-w-4xl">
      <header className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-800">
          OULAD Course Recommender
        </h1>
        <p className="text-gray-600">
          Enter a student ID to view personalized course presentation recommendations.
        </p>
      </header>

      <main>
        <UserSelector onUserSelect={handleUserSelect} isLoading={isLoading} />

        {isLoading && <LoadingSpinner />}
        {error && <ErrorMessage message={error} />}

        {!isLoading && !error && (
          <RecommendationList
             recommendations={recommendations}
             selectedUserId={selectedUserId}
           />
        )}
      </main>

      <footer className="text-center mt-12 text-gray-500 text-sm">
          Powered by ItemCF Model
      </footer>
    </div>
  );
}

export default App;