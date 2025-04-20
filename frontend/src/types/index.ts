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

// --- ADDED: Type definition for Model Info ---
export interface ModelInfo {
    id: string; // e.g., 'itemcf', 'popularity'
    name: string; // e.g., 'Item-Based CF', 'Popularity Baseline'
    description: string;
    pros: string[];
    cons: string[];
}

// --- ADDED: Example data (consider moving to its own file like src/data/modelInfo.ts) ---
export const modelInfos: ModelInfo[] = [
    {
        id: 'popularity',
        name: 'Popularity Baseline',
        description: 'Recommends the most interacted-with courses across all users. Non-personalized.',
        pros: ['Simple to implement', 'No cold-start problem for items', 'Good starting point'],
        cons: ['Not personalized', 'Lower accuracy', 'Can create filter bubbles (popular items get more popular)'],
    },
    {
        id: 'itemcf',
        name: 'Item-Based CF (Demo Model)',
        description: 'Recommends courses similar to those a user previously interacted with, based on co-interaction patterns across all users.',
        pros: ['Personalized', 'Often effective for item discovery', 'Explainable (based on similar items)', 'Relatively efficient after pre-computation'],
        cons: ['Data sparsity can be an issue', 'Cold-start problem for new users', 'Sensitive to interaction data quality', 'Limited by items seen in training'],
    },
    {
        id: 'als',
        name: 'ALS (Matrix Factorization)',
        description: 'Uses Alternating Least Squares to decompose the user-item interaction matrix into latent factors, capturing underlying preferences.',
        pros: ['Personalized', 'Can uncover latent features', 'Scalable (with libraries like implicit)'],
        cons: ['Less interpretable than ItemCF', 'Requires hyperparameter tuning (factors, regularization)', 'Can suffer from cold-start'],
    },
    {
        id: 'ncf',
        name: 'Neural Collaborative Filtering',
        description: 'Uses neural networks (GMF + MLP) to model complex user-item interactions, potentially capturing non-linear relationships.',
        pros: ['Personalized', 'Can model complex patterns', 'Flexible architecture'],
        cons: ['Computationally more expensive to train', 'Requires careful tuning (network structure, learning rate)', 'Can be harder to interpret', 'Prone to overfitting'],
    },
    {
        id: 'hybrid',
        name: 'Hybrid NCF (Content + CF)',
        description: 'Combines collaborative filtering signals (like NCF) with item content features (like course length, VLE activity types) using neural networks.',
        pros: ['Personalized', 'Can leverage item metadata', 'May improve recommendations for less popular items', 'Potentially mitigates some cold-start issues for items'],
        cons: ['Increases model complexity', 'Feature engineering is crucial', 'Requires more data (item features)', 'Tuning becomes more complex'],
    },
];