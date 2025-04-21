// frontend/src/types/index.ts

export interface RecommendationItem {
    presentation_id: string;
    score: number;
    module_id: string;
    presentation_code: string;
    module_presentation_length?: number;
  }

  export interface RecommendationResponse { // Primarily for ensemble result now
    recommendations: RecommendationItem[];
  }

  // Type for the response from the /all_models endpoint
  export type AllModelsRecs = {
      // Key is the model name (string), value is the list of recommendations
      [modelName: string]: RecommendationItem[];
  };


  export interface User {
    student_id: number;
  }

  export interface Presentation {
      presentation_id: string;
      module_id: string;
      presentation_code: string;
      module_presentation_length?: number;
  }

  export interface ModelInfo {
      id: string;
      name: string; // This name should match the keys in AllModelsRecs and MODEL_WEIGHTS
      description: string;
      pros: string[];
      cons: string[];
      evaluationScore?: number; // Optional: Store NDCG or other metric here
  }

  // Ensure model names here match the keys returned by the API (defined in model_loader.py)
  export const modelInfos: ModelInfo[] = [
      {
          id: 'popularity',
          name: 'Popularity', // Matches key
          description: 'Recommends the most interacted-with courses across all users. Non-personalized.',
          pros: ['Simple to implement', 'No cold-start problem for items', 'Good starting point'],
          cons: ['Not personalized', 'Lower accuracy', 'Can create filter bubbles'],
          // evaluationScore: 0.2153 // Example: NDCG@10 from report
      },
      {
          id: 'itemcf',
          name: 'ItemCF', // Matches key
          description: 'Recommends courses similar to those a user previously interacted with, based on co-interaction patterns. Highly weighted in the ensemble.',
          pros: ['Personalized', 'Often effective for item discovery', 'Explainable (based on similar items)', 'Highly effective on this dataset'],
          cons: ['Data sparsity can be an issue', 'Cold-start problem for new users', 'Limited by items seen in training'],
           evaluationScore: 0.6153 // Example: NDCG@10 from report
      },
      {
          id: 'als',
          name: 'ALS (f=100)', // Matches key
          description: 'Uses Alternating Least Squares to decompose the user-item interaction matrix into latent factors.',
          pros: ['Personalized', 'Can uncover latent features', 'Scalable (with libraries like implicit)'],
          cons: ['Less interpretable than ItemCF', 'Requires hyperparameter tuning', 'Can suffer from cold-start', 'Lower performance than ItemCF/NCF here'],
           evaluationScore: 0.3844 // Example: NDCG@10 from report
      },
      {
          id: 'ncf',
          name: 'NCF (e=15)', // Matches key
          description: 'Uses neural networks (GMF + MLP) to model complex user-item interactions.',
          pros: ['Personalized', 'Can model complex patterns', 'Flexible architecture', 'Good performance'],
          cons: ['Computationally expensive', 'Requires careful tuning', 'Harder to interpret', 'Prone to overfitting'],
           evaluationScore: 0.5855 // Example: NDCG@10 from report
      },
      {
          id: 'hybrid',
          name: 'Hybrid (e=15)', // Matches key
          description: 'Combines collaborative filtering signals (NCF) with item content features using neural networks.',
          pros: ['Personalized', 'Can leverage item metadata', 'May improve for less popular items'],
          cons: ['Increases model complexity', 'Feature engineering crucial', 'Tuning complex', 'Lower NDCG than ItemCF/NCF in eval'],
           evaluationScore: 0.4698 // Example: NDCG@10 from report (note lower neg samples)
      },
  ];

  // Helper to find ModelInfo by name
  export const findModelInfoByName = (name: string): ModelInfo | undefined => {
      return modelInfos.find(m => m.name === name);
  };

// --- NEW TYPE FOR PRESENTATION DETAIL MODAL ---
// Can just reuse RecommendationItem for now, as it contains all needed info
export type PresentationDetailInfo = RecommendationItem;