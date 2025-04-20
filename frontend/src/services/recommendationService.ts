// frontend/src/services/recommendationService.ts
import { RecommendationResponse, User, RecommendationItem } from "../types";

// Use environment variable for API URL if available, otherwise default
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api/v1";
console.log("API Base URL:", API_BASE_URL); // Log the URL being used

/**
 * Searches for users based on a query string.
 * @param query - The search string (e.g., the start of a student ID).
 * @param limit - The maximum number of users to return.
 */
export const searchUsers = async (query: string, limit: number = 50): Promise<User[]> => {
  try {
    // Construct the URL with query parameters
    const url = new URL(`${API_BASE_URL}/users`);
    // Only add search param if query is not empty
    if (query) {
        url.searchParams.append('search', query);
    }
    url.searchParams.append('limit', String(limit));

    const response = await fetch(url.toString()); // Use the constructed URL

    if (!response.ok) {
      console.error(`HTTP error searching users! status: ${response.status}`);
      try {
        const errorBody = await response.json();
        throw new Error(errorBody.detail || `HTTP error! status: ${response.status}`);
      } catch {
        throw new Error(`HTTP error searching users! status: ${response.status}`);
      }
    }
    const users: User[] = await response.json();
    return users;
  } catch (error) {
    console.error("Error searching users:", error);
    throw error; // Re-throw the error for the component to handle
  }
};

/**
 * Fetches recommendations for a specific user ID.
 * @param userId - The ID of the student.
 * @param k - The number of recommendations to fetch.
 */
export const fetchRecommendations = async (userId: number, k: number = 10): Promise<RecommendationItem[]> => {
  try {
    const response = await fetch(`${API_BASE_URL}/recommendations/${userId}?k=${k}`);

    if (!response.ok) {
      // Consider 404 as "user not found or no recommendations" -> return empty array
      if (response.status === 404) {
        console.warn(`Recommendations endpoint returned 404 for user ${userId}. Likely unknown user or no recommendations.`);
        return []; // Return empty list gracefully
      }
      // Handle other errors
      console.error(`HTTP error fetching recommendations! status: ${response.status}`);
       try {
            const errorBody = await response.json();
            throw new Error(errorBody.detail || `HTTP error! status: ${response.status}`);
        } catch {
             throw new Error(`HTTP error fetching recommendations! status: ${response.status}`);
        }
    }
    const data: RecommendationResponse = await response.json();
    return data.recommendations || []; // Ensure we return an array
  } catch (error) {
    console.error(`Error fetching recommendations for user ${userId}:`, error);
    throw error; // Re-throw error for component to handle
  }
};

/**
 * Fetches a single random user from the API.
 */
export const fetchRandomUser = async (): Promise<User | null> => {
    try {
      const response = await fetch(`${API_BASE_URL}/users/random`);

      if (!response.ok) {
        // Handle 404 specifically (no users in dataset)
        if (response.status === 404) {
          console.warn("API reported no users found for random selection.");
          return null;
        }
        // Handle other errors
        console.error(`HTTP error fetching random user! status: ${response.status}`);
        try {
          const errorBody = await response.json();
          throw new Error(errorBody.detail || `HTTP error! status: ${response.status}`);
        } catch {
          throw new Error(`HTTP error fetching random user! status: ${response.status}`);
        }
      }
      // API returns a single user object or null
      const user: User | null = await response.json();
      return user;
    } catch (error) {
      console.error("Error fetching random user:", error);
      throw error; // Re-throw error for component to handle
    }
  };