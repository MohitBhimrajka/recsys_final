// frontend/src/services/recommendationService.ts
import { RecommendationResponse, User, RecommendationItem } from "../types";

const API_BASE_URL = "http://localhost:8000/api/v1"; // Ensure this matches your API

/**
 * Fetches the list of available users from the API.
 */
export const fetchUsers = async (): Promise<User[]> => {
  try {
    const response = await fetch(`${API_BASE_URL}/users`);
    if (!response.ok) {
        console.error(`HTTP error fetching users! status: ${response.status}`);
        // Attempt to read error response body if available
        try {
            const errorBody = await response.json();
            throw new Error(errorBody.detail || `HTTP error! status: ${response.status}`);
        } catch {
             throw new Error(`HTTP error! status: ${response.status}`);
        }
    }
    const users: User[] = await response.json();
    return users.sort((a, b) => a.student_id - b.student_id);
  } catch (error) {
    console.error("Error fetching users:", error);
    throw error; // Re-throw the error so the component can catch it
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
      // Handle 404 specifically - user might exist but have no recs, or unknown user
      if (response.status === 404) {
        // API doesn't distinguish between "unknown user" and "known user with no recs"
        // It just doesn't find the user in the model. The service layer returns empty array.
        console.warn(`User ${userId} not found in model or no recommendations available.`);
        return []; // Return empty list gracefully
      }
      // Handle other errors
      console.error(`HTTP error fetching recommendations! status: ${response.status}`);
       try {
            const errorBody = await response.json();
            throw new Error(errorBody.detail || `HTTP error! status: ${response.status}`);
        } catch {
             throw new Error(`HTTP error! status: ${response.status}`);
        }
    }
    const data: RecommendationResponse = await response.json();
    return data.recommendations || []; // Ensure we return an array
  } catch (error) {
    console.error(`Error fetching recommendations for user ${userId}:`, error);
    throw error; // Re-throw error for component to handle
  }
};