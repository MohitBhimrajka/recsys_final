// frontend/src/services/recommendationService.ts
import { RecommendationResponse, User, RecommendationItem, AllModelsRecs } from "../types"; // Import new type

// Use environment variable for API URL if available, otherwise default
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api/v1";
console.log("API Base URL:", API_BASE_URL);

/**
 * Searches for users based on a query string.
 * @param query - The search string (e.g., the start of a student ID).
 * @param limit - The maximum number of users to return.
 */
export const searchUsers = async (query: string, limit: number = 50): Promise<User[]> => {
  try {
    const url = new URL(`${API_BASE_URL}/users`);
    if (query) {
        url.searchParams.append('search', query);
    }
    url.searchParams.append('limit', String(limit));
    const response = await fetch(url.toString());
    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({ detail: `HTTP error searching users! status: ${response.status}` }));
      throw new Error(errorBody.detail || `HTTP error searching users! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error searching users:", error);
    throw error;
  }
};

/**
 * Fetches ENSEMBLE recommendations for a specific user ID.
 * @param userId - The ID of the student.
 * @param k - The number of recommendations to fetch.
 */
export const fetchRecommendations = async (userId: number, k: number = 10): Promise<RecommendationItem[]> => {
  try {
    const response = await fetch(`${API_BASE_URL}/recommendations/${userId}?k=${k}`);
    if (!response.ok) {
      if (response.status === 404) {
        console.warn(`Ensemble endpoint returned 404 for user ${userId}.`);
        return []; // Return empty list gracefully
      }
      const errorBody = await response.json().catch(() => ({ detail: `HTTP error fetching ensemble recommendations! status: ${response.status}` }));
      throw new Error(errorBody.detail || `HTTP error fetching ensemble recommendations! status: ${response.status}`);
    }
    const data: RecommendationResponse = await response.json();
    return data.recommendations || [];
  } catch (error) {
    console.error(`Error fetching ENSEMBLE recommendations for user ${userId}:`, error);
    throw error;
  }
};

/**
 * Fetches recommendations from ALL individual models for a specific user ID.
 * @param userId - The ID of the student.
 * @param k - The number of recommendations per model to fetch.
 */
export const fetchAllModelRecommendations = async (userId: number, k: number = 10): Promise<AllModelsRecs> => {
    try {
        // Note the new endpoint URL
        const response = await fetch(`${API_BASE_URL}/recommendations/${userId}/all_models?k=${k}`);
        if (!response.ok) {
            if (response.status === 404) {
                 console.warn(`All models endpoint returned 404 for user ${userId}.`);
                 // Return an empty object, the component can handle this
                 return {};
            }
            const errorBody = await response.json().catch(() => ({ detail: `HTTP error fetching all model recommendations! status: ${response.status}` }));
            throw new Error(errorBody.detail || `HTTP error fetching all model recommendations! status: ${response.status}`);
        }
        // The response schema has a 'results' key containing the dictionary
        const data: { results: AllModelsRecs } = await response.json();
        return data.results || {}; // Return the dictionary inside 'results' or empty object
    } catch (error) {
        console.error(`Error fetching all model recommendations for user ${userId}:`, error);
        throw error;
    }
};


/**
 * Fetches a single random user from the API.
 */
export const fetchRandomUser = async (): Promise<User | null> => {
    try {
      const response = await fetch(`${API_BASE_URL}/users/random`);
      if (!response.ok) {
        if (response.status === 404) {
          console.warn("API reported no users found for random selection.");
          return null;
        }
        const errorBody = await response.json().catch(() => ({ detail: `HTTP error fetching random user! status: ${response.status}` }));
        throw new Error(errorBody.detail || `HTTP error fetching random user! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error("Error fetching random user:", error);
      throw error;
    }
  };