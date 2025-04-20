// frontend/src/types/index.ts

export interface RecommendationItem {
  presentation_id: string;
  score: number;
  module_id: string;
  presentation_code: string;
  module_presentation_length?: number; // Optional property
}

// The API returns an object with a 'recommendations' key
export interface RecommendationResponse {
  recommendations: RecommendationItem[];
}

export interface User {
  student_id: number;
}

// Optional, if you fetch presentation details
export interface Presentation {
    presentation_id: string;
    module_id: string;
    presentation_code: string;
    module_presentation_length?: number;
}