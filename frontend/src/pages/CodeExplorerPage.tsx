// frontend/src/pages/CodeExplorerPage.tsx
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { FiCode, FiDatabase, FiLayout, FiBarChart, FiZap, FiSettings, FiCpu, FiFilter, FiBox, FiInfo, FiMinusSquare, FiPlusSquare, FiExternalLink } from 'react-icons/fi';
import TreeNodeComponent from '../components/TreeNodeComponent';
import CodeHighlightsSection from '../components/CodeHighlightsSection';
import { CodeCategory, TreeNode, CodeItem } from '../types/codeExplorer';
import { optimizedVariants, hardwareAcceleration } from '../utils/animationUtils';

// Code Snippets (Unchanged from Phase 4)
const preprocessSnippet = `
def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    # ... group daily clicks ...
    user_item_interactions = daily_interactions.groupby(['id_student', 'presentation_id']).agg(
        total_clicks=('sum_click', 'sum'),
        interaction_days=('date', 'nunique'),
        # ... other aggregations ...
    ).reset_index()
    # Calculate implicit feedback score (log transformation)
    user_item_interactions['implicit_feedback'] = np.log1p(user_item_interactions['total_clicks'])
    return user_item_interactions
`;
const itemcfSnippet = `
def predict(self, user_id: Any, item_ids: List[Any]) -> List[float]:
    user_idx = self.user_id_to_idx.get(user_id)
    if user_idx is None: return [0.0] * len(item_ids)

    user_interactions_vector = self.interaction_matrix_sparse[user_idx]
    scores = [0.0] * len(item_ids)

    for i, target_id in enumerate(item_ids):
        target_idx = self.item_id_to_idx.get(target_id)
        if target_idx is None: continue

        target_item_similarities = self.item_similarity_matrix[target_idx]
        score = user_interactions_vector.dot(target_item_similarities.T).toarray()[0, 0]
        scores[i] = float(score)
    return scores
`;
const apiServiceSnippet = `
# Simplified ensemble logic example
def get_ensemble_recommendations(...) -> List[schemas.RecommendationItem]:
    # ... get candidates ...
    final_item_scores = defaultdict(float)
    total_weight = sum(MODEL_WEIGHTS.values())

    for model_name, model in models.items():
        # ... get raw scores ...
        # ... calculate normalized_score ...
        weight = MODEL_WEIGHTS.get(model_name, 0) / total_weight
        final_item_scores[item_id] += normalized_score * weight

    sorted_items = sorted(final_item_scores.items(), key=lambda item: item[1], reverse=True)
    top_k_ensemble = sorted_items[:k]
    # ... format results ...
    return results
`;
const evaluatorSnippet = `
def evaluate_model(self, model: BaseRecommender, n_neg_samples: Optional[int] = None) -> Dict[str, float]:
    # ... loop through test users ...
    for user_id in tqdm(test_users_known_by_model, desc="Evaluating users"):
        test_positives_known = # ... get known relevant items ...
        if not test_positives_known: continue

        # --- Determine items_to_predict (negative sampling or full) ---
        if n_neg_samples is not None:
             # ... sample negatives ...
            items_to_predict = test_positives_known + sampled_negatives
        else: items_to_predict = list(model_known_items - known_positives)
        if not items_to_predict: continue

        scores = model.predict(user_id, items_to_predict)
        # ... rank items ... get top_k_recs ...
        # --- Calculate metrics (P@k, R@k, NDCG@k) ---
        prec = precision_at_k(top_k_recs, test_positives_known, self.k)
        # ... calculate recall, ndcg ... store results ...
    # --- Aggregate results ---
    return { ... metrics ... }
`;
const baseModelSnippet = `
class BaseRecommender(ABC):
    """Abstract base class for all recommendation models."""
    
    @abstractmethod
    def fit(self, interactions_df: pd.DataFrame, user_features_df: Optional[pd.DataFrame] = None, 
            item_features_df: Optional[pd.DataFrame] = None) -> None:
        """Train the model on the provided data."""
        pass
        
    @abstractmethod
    def predict(self, user_id: Any, item_ids: List[Any]) -> List[float]:
        """
        Predict the score/relevance of items for the given user.
        
        Args:
            user_id: Identifier for the user
            item_ids: List of item identifiers to predict scores for
            
        Returns:
            List of prediction scores in the same order as item_ids
        """
        pass
        
    @abstractmethod
    def get_known_users(self) -> Set[Any]:
        """Return set of user IDs the model can make predictions for."""
        pass
        
    @abstractmethod
    def get_known_items(self) -> Set[Any]:
        """Return set of item IDs the model can make predictions for."""
        pass
`;
const reactComponentSnippet = `
// RecommendationCard.tsx
const RecommendationCard: React.FC<RecommendationCardProps> = ({ 
  recommendation, modelName, isHighlighted = false 
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const [showDetails, setShowDetails] = useState(false);
  
  return (
    <motion.div
      className={\`bg-surface p-4 rounded-xl border shadow-sm transition \${
        isHighlighted 
          ? 'border-primary/50 shadow-primary/10'
          : 'border-border-color hover:border-primary/30'
      }\`}
      whileHover={{ y: -5, transition: { duration: 0.2 } }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className="flex items-start justify-between">
        <h3 className="text-lg font-medium text-text-primary">
          {recommendation.title}
        </h3>
        <ModelTag model={modelName} size="sm" />
      </div>
      
      {/* Score & Description */}
      <div className="mt-2 flex items-center">
        <ScoreIndicator score={recommendation.score} />
        <p className="text-sm text-text-muted ml-3 line-clamp-1">
          {recommendation.description}
        </p>
      </div>
      
      {/* Actions */}
      <motion.div 
        className="mt-3 flex justify-between items-center"
        initial={{ opacity: 0 }}
        animate={{ opacity: isHovered ? 1 : 0 }}
      >
        <button 
          onClick={() => setShowDetails(true)}
          className="text-xs text-primary hover:underline"
        >
          View Details
        </button>
      </motion.div>
    </motion.div>
  );
};
`;

// Project Tree Data (Unchanged from Phase 4)
const addPaths = (node: any, currentPath: string = ''): any => { /* ... implementation unchanged ... */
    const nodePath = currentPath ? `${currentPath}/${node.name}` : node.name;
    const newNode = { ...node, path: nodePath };
    if (node.children) { newNode.children = node.children.map((child: any) => addPaths(child, nodePath)); }
    return newNode;
 };
// REPLACE the existing rawTreeData object with this corrected and complete version:
const rawTreeData = {
    name: 'mohitbhimrajka-recsys_final/', type: 'folder', comment: 'Project Root', children: [
        // Root Files
        { name: 'README.md', type: 'file', comment: 'Project overview, setup, usage' },
        { name: 'pytest.ini', type: 'file', comment: 'Pytest configuration' },
        { name: 'requirements.txt', type: 'file', comment: 'Core Python dependencies' },
        { name: '.env.example', type: 'file', comment: 'Database credential template' },
        // api/ Directory
        { name: 'api/', type: 'folder', comment: 'FastAPI Backend Server', children: [
            { name: 'requirements.txt', type: 'file', comment: 'API specific dependencies' },
            { name: 'app/', type: 'folder', comment: 'FastAPI application code', children: [
                { name: '__init__.py', type: 'file' },
                { name: 'main.py', type: 'file', comment: 'App setup, CORS, startup events' },
                { name: 'model_loader.py', type: 'file', comment: 'Loads trained models & data on startup' },
                { name: 'schemas.py', type: 'file', comment: 'Pydantic API request/response models' },
                { name: 'services.py', type: 'file', comment: 'Core recommendation & metadata logic' },
                { name: 'routers/', type: 'folder', comment: 'API endpoint definitions', children: [
                    { name: '__init__.py', type: 'file' },
                    { name: 'recommendations.py', type: 'file', comment: 'Endpoints for recs, users, etc.' }
                ]}
            ]}
        ]},
        // frontend/ Directory
        { name: 'frontend/', type: 'folder', comment: 'React/Vite Frontend UI', children: [
            { name: 'README.md', type: 'file', comment: 'Frontend specific info (Vite default)' },
            { name: 'eslint.config.js', type: 'file', comment: 'ESLint configuration' },
            { name: 'index.html', type: 'file', comment: 'Main HTML entry point' },
            { name: 'postcss.config.cjs', type: 'file', comment: 'PostCSS configuration' },
            { name: 'tailwind.config.js', type: 'file', comment: 'Tailwind CSS theme configuration' },
            { name: 'vite.config.ts', type: 'file', comment: 'Vite build tool configuration' },
            { name: '.gitignore', type: 'file', comment: 'Specifies ignored files for Git' },
            // { name: 'package.json', type: 'file', comment: 'NPM dependencies and scripts' }, // Add if it exists
            // { name: 'pnpm-lock.yaml', type: 'file', comment: 'PNPM lock file'}, // Add if it exists
            { name: 'public/', type: 'folder', comment: 'Static assets', children: [
                { name: 'icons/', type: 'folder', comment: 'SVG/Icon files' } // Assuming icons folder exists based on prompt structure
            ]},
            { name: 'src/', type: 'folder', comment: 'Frontend source code', children: [
                { name: 'App.css', type: 'file', comment: 'Minimal base CSS (mostly Tailwind)' },
                { name: 'App.tsx', type: 'file', comment: 'Root component, routing setup' },
                { name: 'index.css', type: 'file', comment: 'Tailwind directives, global styles' },
                { name: 'main.tsx', type: 'file', comment: 'Application entry point' },
                { name: 'vite-env.d.ts', type: 'file', comment: 'Vite TypeScript env types' },
                { name: 'components/', type: 'folder', comment: 'Reusable UI components', children: [
                    { name: 'AnalysisCard.tsx', type: 'file' },
                    { name: 'AnalysisDashboard.tsx', type: 'file' },
                    { name: 'ErrorMessage.tsx', type: 'file' },
                    { name: 'FeatureCard.tsx', type: 'file' },
                    { name: 'Footer.tsx', type: 'file' },
                    { name: 'HighlightItem.tsx', type: 'file' },
                    { name: 'Layout.tsx', type: 'file' },
                    { name: 'LoadingSpinner.tsx', type: 'file' },
                    { name: 'ModelInfoModal.tsx', type: 'file' },
                    { name: 'ModelTag.tsx', type: 'file' },
                    { name: 'Navbar.tsx', type: 'file' },
                    { name: 'OverlapChart.tsx', type: 'file' },
                    { name: 'PresentationDetailModal.tsx', type: 'file' },
                    { name: 'ProcessStep.tsx', type: 'file' },
                    { name: 'RankComparisonTable.tsx', type: 'file' },
                    { name: 'RankCorrelationDisplay.tsx', type: 'file' },
                    { name: 'RecommendationCard.tsx', type: 'file' },
                    { name: 'RecommendationConsensus.tsx', type: 'file' },
                    { name: 'RecommendationList.tsx', type: 'file' },
                    { name: 'ScrollToTop.tsx', type: 'file' },
                    { name: 'SkeletonCard.tsx', type: 'file' },
                    { name: 'Tooltip.tsx', type: 'file' },
                    { name: 'TreeNodeComponent.tsx', type: 'file' },
                    { name: 'UserSelector.tsx', type: 'file' }
                ]},
                { name: 'pages/', type: 'folder', comment: 'Top-level page components', children: [
                    { name: 'AboutPage.tsx', type: 'file' },
                    { name: 'CodeExplorerPage.tsx', type: 'file' },
                    { name: 'DemoPage.tsx', type: 'file' },
                    { name: 'HomePage.tsx', type: 'file' }
                ]},
                { name: 'services/', type: 'folder', comment: 'API interaction functions', children: [
                    { name: 'recommendationService.ts', type: 'file' }
                ]},
                { name: 'types/', type: 'folder', comment: 'TypeScript type definitions', children: [
                    { name: 'index.ts', type: 'file' },
                    { name: 'spearman-rho.d.ts', type: 'file' }
                ]}
            ]}
        ]},
        // notebooks/ Directory
        { name: 'notebooks/', type: 'folder', comment: 'Jupyter Dev & Exploration', children: [
            { name: '01_eda.ipynb', type: 'file', comment: 'Exploratory Data Analysis' },
            { name: '02_preprocessing_feature_eng.ipynb', type: 'file', comment: 'Preprocessing steps dev'},
            { name: '03_baseline_models.ipynb', type: 'file', comment: 'Popularity, ItemCF, ALS dev'},
            { name: '04_ncf_dev.ipynb', type: 'file', comment: 'NCF model development' },
            { name: '05_content_hybrid_dev.ipynb', type: 'file', comment: 'Hybrid model development' },
        ]},
        // reports/ Directory
        { name: 'reports/', type: 'folder', comment: 'Project reports (Markdown)', children: [
            { name: 'eda_summary.md', type: 'file', comment: 'Summary of EDA findings' },
            { name: 'final_report.md', type: 'file', comment: 'Final project report' },
            { name: 'interim_report.md', type: 'file', comment: 'Mid-project status report' }
        ]},
        // src/ Directory (Core Python Library)
        { name: 'src/', type: 'folder', comment: 'Core Python Library (Backend Logic)', children: [
            { name: 'config.py', type: 'file', comment: 'Configuration (paths, params, DB URI)' },
            { name: 'data/', type: 'folder', comment: 'Data loading & processing modules', children: [
                { name: 'dataset.py', type: 'file', comment: 'PyTorch Dataset classes (CF, Hybrid)' },
                { name: 'load_raw.py', type: 'file', comment: 'Functions to load raw CSVs' },
                { name: 'preprocess.py', type: 'file', comment: 'Data cleaning & feature engineering' },
                { name: 'utils.py', type: 'file', comment: 'Helper functions (ID creation, mappings)' },
            ]},
            { name: 'database/', type: 'folder', comment: 'DB schema & loading scripts (Optional)', children: [
                { name: 'db_utils.py', type: 'file', comment: 'DB connection helpers' },
                { name: 'load_to_db.py', type: 'file', comment: 'Script to load processed data' },
                { name: 'schema.py', type: 'file', comment: 'SQLAlchemy table definitions' },
            ]},
            { name: 'evaluation/', type: 'folder', comment: 'Model evaluation framework', children: [
                { name: 'evaluator.py', type: 'file', comment: 'RecEvaluator class' },
                { name: 'metrics.py', type: 'file', comment: 'Precision@k, Recall@k, NDCG@k' }
            ]},
            { name: 'models/', type: 'folder', comment: 'Recommender model implementations', children: [
                { name: 'base.py', type: 'file', comment: 'Abstract BaseRecommender class' },
                { name: 'content_encoder.py', type: 'file', comment: 'MLP for item features (Hybrid)' },
                { name: 'hybrid.py', type: 'file', comment: 'Hybrid NCF model & wrapper' },
                { name: 'item_cf.py', type: 'file', comment: 'Item-based CF model' },
                { name: 'matrix_factorization.py', type: 'file', comment: 'Implicit ALS wrapper' },
                { name: 'ncf.py', type: 'file', comment: 'NCF model & wrapper' },
                { name: 'popularity.py', type: 'file', comment: 'Popularity baseline model' },
            ]},
            { name: 'pipelines/', type: 'folder', comment: 'End-to-end execution scripts', children: [
                { name: 'evaluate.py', type: 'file', comment: 'Evaluate a saved model' },
                { name: 'run_preprocessing.py', type: 'file', comment: 'Execute full preprocessing' },
                { name: 'setup_database.py', type: 'file', comment: 'Create/drop DB tables (Optional)' },
                { name: 'train.py', type: 'file', comment: 'Train a specified model' },
            ]},
        ]},
        // tests/ Directory
        { name: 'tests/', type: 'folder', comment: 'Pytest unit/integration tests', children: [
             { name: 'test_database.py', type: 'file' },
             { name: 'test_evaluation.py', type: 'file' },
             { name: 'test_models.py', type: 'file' },
             { name: 'test_preprocessing.py', type: 'file' },
             { name: 'test_utils.py', type: 'file' },
        ]},
    ]
};
const projectTreeData = addPaths(rawTreeData);
const GITHUB_BASE_URL = "https://github.com/mohitbhimrajka/recsys_final/blob/main";

// Define page animation variant
const pageVariant = {
  hidden: { opacity: 0, y: 15 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: "easeOut" } },
};

// Code highlights data for the enhanced section
const codeHighlights: CodeItem[] = [
  {
    id: 'data-preprocessing',
    icon: <FiFilter />,
    title: 'Data Preprocessing',
    path: 'src/data/preprocess.py',
    githubUrl: `${GITHUB_BASE_URL}/src/data/preprocess.py`,
    description: 'Cleans raw OULAD data, filters interactions based on registration periods and activity thresholds (e.g., min 5 interactions/users), aggregates VLE clicks into an <code>implicit_feedback</code> score using <code>log1p</code>, and generates final user/item feature Parquet files.',
    codeSnippet: preprocessSnippet,
    language: 'python',
    category: ['python', 'data', 'backend'] as CodeCategory[],
    isFeatured: true
  },
  {
    id: 'itemcf-model',
    icon: <FiCpu />,
    title: 'ItemCF Model Logic',
    path: 'src/models/item_cf.py',
    githubUrl: `${GITHUB_BASE_URL}/src/models/item_cf.py`,
    description: 'Item-Based Collaborative Filtering logic. Computes a cosine similarity matrix between courses based on co-interaction patterns. Predictions sum similarities weighted by the user\'s past interaction scores. This model performed best in offline tests.',
    codeSnippet: itemcfSnippet,
    language: 'python',
    category: ['python', 'models', 'backend'] as CodeCategory[]
  },
  {
    id: 'api-service',
    icon: <FiZap />,
    title: 'API Service Logic',
    path: 'api/app/services.py',
    githubUrl: `${GITHUB_BASE_URL}/api/app/services.py`,
    description: 'Core logic for the FastAPI backend. Includes functions to get candidate items, predict scores using loaded models (like <code>model.predict(user_id, items)</code>), calculate weighted ensemble scores, and format results using Pydantic schemas.',
    codeSnippet: apiServiceSnippet,
    language: 'python',
    category: ['python', 'backend'] as CodeCategory[]
  },
  {
    id: 'evaluation-framework',
    icon: <FiBarChart />,
    title: 'Evaluation Framework',
    path: 'src/evaluation/evaluator.py',
    githubUrl: `${GITHUB_BASE_URL}/src/evaluation/evaluator.py`,
    description: 'Implements evaluation using a time-based split. The <code>RecEvaluator</code> class calculates ranking metrics (Precision@k, Recall@k, NDCG@k) comparing model predictions against ground truth. Supports negative sampling for efficiency.',
    codeSnippet: evaluatorSnippet,
    language: 'python',
    category: ['python', 'evaluation', 'backend'] as CodeCategory[]
  },
  {
    id: 'frontend-ui',
    icon: <FiLayout />,
    title: 'Frontend UI (React)',
    path: 'frontend/src/components/RecommendationCard.tsx',
    githubUrl: `${GITHUB_BASE_URL}/frontend/src/components/RecommendationCard.tsx`,
    description: 'This interactive UI component displays a recommendation with score, title and model source. Built using React, TypeScript, and Tailwind CSS with Framer Motion animations. Demonstrates interactive hover effects and conditional styling.',
    codeSnippet: reactComponentSnippet,
    language: 'typescript',
    category: ['typescript', 'frontend'] as CodeCategory[]
  },
  {
    id: 'model-abstraction',
    icon: <FiBox />,
    title: 'Model Abstraction',
    path: 'src/models/base.py',
    githubUrl: `${GITHUB_BASE_URL}/src/models/base.py`,
    description: 'Defines the <code>BaseRecommender</code> abstract class. All models (Popularity, ItemCF, ALS, NCF, Hybrid) inherit from it, ensuring a consistent interface with <code>fit()</code>, <code>predict()</code>, <code>get_known_users()</code>, and <code>get_known_items()</code> methods, crucial for standardized evaluation and API usage.',
    codeSnippet: baseModelSnippet,
    language: 'python',
    category: ['python', 'models', 'backend'] as CodeCategory[]
  }
];

// Wrapper Component for Tree View (Now defaults to expanded)
const InteractiveDirectoryTree: React.FC<{ structure: TreeNode }> = ({ structure }) => {
    const [searchTerm, setSearchTerm] = useState('');
    const [allExpanded, setAllExpanded] = useState(false);
    
    const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setSearchTerm(event.target.value);
        if (event.target.value !== '') {
            setAllExpanded(true);
        }
    };
    
    const toggleAll = () => setAllExpanded(!allExpanded);
    
    const githubBaseUrl = 'https://github.com/username/recsys_final';
    
    return (
        <div className="bg-surface border border-border-color/50 rounded-lg p-4 shadow-sm">
            <div className="mb-4 flex flex-wrap gap-3 items-center justify-between">
                <h3 className="text-lg font-medium text-text-primary">Project Structure</h3>
                <div className="flex gap-2">
                    <button 
                        className="text-xs px-3 py-1.5 rounded-md flex items-center gap-1.5 bg-primary/10 text-primary hover:bg-primary/20 transition-colors"
                        onClick={toggleAll}
                    >
                        {allExpanded ? <FiMinusSquare size={14} /> : <FiPlusSquare size={14} />}
                        {allExpanded ? 'Collapse All' : 'Expand All'}
                    </button>
                    <a 
                        href={githubBaseUrl} 
                        target="_blank" 
                        rel="noopener noreferrer" 
                        className="text-xs px-3 py-1.5 rounded-md flex items-center gap-1.5 bg-background text-text-muted hover:bg-border-color/30 transition-colors"
                    >
                        <FiExternalLink size={14} />
                        View on GitHub
                    </a>
                </div>
            </div>
            
            <div className="mb-4">
                <input 
                    type="text" 
                    placeholder="Search files..." 
                    value={searchTerm} 
                    onChange={handleSearchChange} 
                    className="w-full px-4 py-2 bg-background border border-border-color rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary"
                />
            </div>
            
            <div className="overflow-auto max-h-[70vh]">
                <TreeNodeComponent 
                    node={structure} 
                    level={0} 
                    githubBaseUrl={githubBaseUrl} 
                    searchTerm={searchTerm} 
                    allExpanded={allExpanded} 
                    filterVisible={true} 
                />
            </div>
        </div>
    );
};

// --- Main CodeExplorerPage Component ---
const CodeExplorerPage: React.FC = () => {
    return (
        <motion.div
            className="py-16 md:py-24"
            variants={pageVariant}
            initial="hidden"
            animate="visible"
        >
            <motion.h1 className="text-4xl md:text-5xl font-bold text-center mb-16 md:mb-20 text-text-primary" initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
                Project Structure & Code Highlights
            </motion.h1>

            {/* Interactive Directory Structure */}
            <motion.div className="mb-20 md:mb-24" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5, delay: 0.1 }}>
                <h2 className="text-2xl md:text-3xl lg:text-4xl font-semibold text-text-primary mb-4">Directory Overview</h2>
                {/* Added Explanatory Text */}
                 <p className="text-text-muted text-xs mb-6 flex items-center gap-2">
                     <FiInfo size={16} className="flex-shrink-0"/>
                     Click folders <FiMinusSquare size={12} className="inline-block mx-0.5"/> / <FiPlusSquare size={12} className="inline-block mx-0.5"/> to toggle, hover for details/paths, search, or click <FiExternalLink size={12} className="inline-block mx-0.5"/> to view on GitHub.
                 </p>
                <InteractiveDirectoryTree structure={projectTreeData} />
            </motion.div>

            {/* Enhanced Code Highlights Section */}
            <motion.div 
                className="mb-16" 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5, delay: 0.2 }}
            >
                <CodeHighlightsSection 
                    codeItems={codeHighlights} 
                    githubBaseUrl={GITHUB_BASE_URL} 
                />
            </motion.div>
        </motion.div>
    );
};

export default CodeExplorerPage;