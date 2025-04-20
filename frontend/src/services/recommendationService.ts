// frontend/src/services/recommendationService.ts
import { RecommendationResponse, User, RecommendationItem } from "../types"; // Adjust path if needed

// Ensure this matches the running port and prefix of your API
const API_BASE_URL = "http://localhost:8000/api/v1";

/**
 * Fetches the list of available users from the API.
 */
export const fetchUsers = async (): Promise<User[]> => {
  try {
    const response = await fetch(`${API_BASE_URL}/users`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const users: User[] = await response.json();
    // Sort users by student_id for consistency
    return users.sort((a, b) => a.student_id - b.student_id);
  } catch (error) {
    console.error("Error fetching users:", error);
    // Return empty array or re-throw, depending on how you want to handle errors
    return [];
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
        if(response.status === 404) {
            console.warn(`User ${userId} not found or no recommendations available.`);
            return []; // Return empty list for not found or no recs
        }
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data: RecommendationResponse = await response.json();
    return data.recommendations || []; // Ensure we return an array
  } catch (error) {
    console.error(`Error fetching recommendations for user ${userId}:`, error);
    return []; // Return empty array on fetch error
  }
};

// Add fetchPresentations if needed later