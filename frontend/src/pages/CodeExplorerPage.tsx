// frontend/src/pages/CodeExplorerPage.tsx
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { FiCode, FiDatabase, FiLayout, FiBarChart, FiZap, FiSettings, FiCpu, FiFilter, FiBox, FiInfo, FiMinusSquare, FiPlusSquare, FiExternalLink } from 'react-icons/fi';
import TreeNodeComponent from '../components/TreeNodeComponent';
import HighlightItem from '../components/HighlightItem';

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

// Project Tree Data (Unchanged from Phase 4)
const addPaths = (node: any, currentPath: string = ''): any => { /* ... implementation unchanged ... */
    const nodePath = currentPath ? `${currentPath}/${node.name}` : node.name;
    const newNode = { ...node, path: nodePath };
    if (node.children) { newNode.children = node.children.map((child: any) => addPaths(child, nodePath)); }
    return newNode;
 };
const rawTreeData = { /* ... structure unchanged from Phase 4 ... */
    name: 'mohitbhimrajka-recsys_final/', type: 'folder', comment: 'Project Root', children: [
        { name: 'README.md', type: 'file', comment: 'Project overview, setup, usage' },
        { name: 'pytest.ini', type: 'file', comment: 'Pytest configuration' },
        { name: 'requirements.txt', type: 'file', comment: 'Core Python dependencies' },
        { name: '.env.example', type: 'file', comment: 'Database credential template' },
        { name: 'api/', type: 'folder', comment: 'FastAPI Backend Server', children: [ { name: 'requirements.txt', type: 'file', comment: 'API specific dependencies' }, { name: 'app/', type: 'folder', comment: 'FastAPI application code', children: [ { name: '__init__.py', type: 'file' }, { name: 'main.py', type: 'file', comment: 'App setup, CORS, startup events' }, { name: 'model_loader.py', type: 'file', comment: 'Loads trained models & data on startup' }, { name: 'schemas.py', type: 'file', comment: 'Pydantic API request/response models' }, { name: 'services.py', type: 'file', comment: 'Core recommendation & metadata logic' }, { name: 'routers/', type: 'folder', comment: 'API endpoint definitions', children: [ { name: '__init__.py', type: 'file' }, { name: 'recommendations.py', type: 'file', comment: 'Endpoints for recs, users, etc.' } ]} ]} ]},
        { name: 'frontend/', type: 'folder', comment: 'React/Vite Frontend UI', children: [ { name: 'README.md', type: 'file', comment: 'Frontend specific info (Vite default)' }, { name: 'eslint.config.js', type: 'file', comment: 'ESLint configuration' }, { name: 'index.html', type: 'file', comment: 'Main HTML entry point' }, { name: 'postcss.config.cjs', type: 'file', comment: 'PostCSS configuration' }, { name: 'tailwind.config.js', type: 'file', comment: 'Tailwind CSS theme configuration' }, { name: 'vite.config.ts', type: 'file', comment: 'Vite build tool configuration' }, { name: '.gitignore', type: 'file', comment: 'Specifies ignored files for Git' }, { name: 'package.json', type: 'file', comment: 'NPM dependencies and scripts' }, { name: 'public/', type: 'folder', comment: 'Static assets', children: [ ] }, { name: 'src/', type: 'folder', comment: 'Frontend source code', children: [ { name: 'App.css', type: 'file', comment: 'Minimal base CSS (mostly Tailwind)' }, { name: 'App.tsx', type: 'file', comment: 'Root component, routing setup' }, { name: 'index.css', type: 'file', comment: 'Tailwind directives, global styles' }, { name: 'main.tsx', type: 'file', comment: 'Application entry point' }, { name: 'vite-env.d.ts', type: 'file', comment: 'Vite TypeScript env types' }, { name: 'components/', type: 'folder', comment: 'Reusable UI components', children: [ ] }, { name: 'pages/', type: 'folder', comment: 'Top-level page components', children: [ ] }, { name: 'services/', type: 'folder', comment: 'API interaction functions', children: [ { name: 'recommendationService.ts', type: 'file' } ]}, { name: 'types/', type: 'folder', comment: 'TypeScript type definitions', children: [ { name: 'index.ts', type: 'file' } ]} ]} ]},
        { name: 'notebooks/', type: 'folder', comment: 'Jupyter Dev & Exploration', children: [ { name: '01_eda.ipynb', type: 'file', comment: 'Exploratory Data Analysis' }, { name: '02_preprocessing_feature_eng.ipynb', type: 'file', comment: 'Preprocessing steps dev'}, { name: '03_baseline_models.ipynb', type: 'file', comment: 'Popularity, ItemCF, ALS dev'}, { name: '04_ncf_dev.ipynb', type: 'file', comment: 'NCF model development' }, { name: '05_content_hybrid_dev.ipynb', type: 'file', comment: 'Hybrid model development' }, ]},
        { name: 'reports/', type: 'folder', comment: 'Project reports (Markdown)', children: [ ] },
        { name: 'src/', type: 'folder', comment: 'Core Python Library (Backend Logic)', children: [ { name: 'config.py', type: 'file', comment: 'Configuration (paths, params, DB URI)' }, { name: 'data/', type: 'folder', comment: 'Data loading & processing modules', children: [ { name: 'dataset.py', type: 'file', comment: 'PyTorch Dataset classes (CF, Hybrid)' }, { name: 'load_raw.py', type: 'file', comment: 'Functions to load raw CSVs' }, { name: 'preprocess.py', type: 'file', comment: 'Data cleaning & feature engineering' }, { name: 'utils.py', type: 'file', comment: 'Helper functions (ID creation, mappings)' }, ]}, { name: 'database/', type: 'folder', comment: 'DB schema & loading scripts (Optional)', children: [ { name: 'db_utils.py', type: 'file', comment: 'DB connection helpers' }, { name: 'load_to_db.py', type: 'file', comment: 'Script to load processed data' }, { name: 'schema.py', type: 'file', comment: 'SQLAlchemy table definitions' }, ]}, { name: 'evaluation/', type: 'folder', comment: 'Model evaluation framework', children: [ { name: 'evaluator.py', type: 'file', comment: 'RecEvaluator class' }, { name: 'metrics.py', type: 'file', comment: 'Precision@k, Recall@k, NDCG@k' } ]}, { name: 'models/', type: 'folder', comment: 'Recommender model implementations', children: [ { name: 'base.py', type: 'file', comment: 'Abstract BaseRecommender class' }, { name: 'content_encoder.py', type: 'file', comment: 'MLP for item features (Hybrid)' }, { name: 'hybrid.py', type: 'file', comment: 'Hybrid NCF model & wrapper' }, { name: 'item_cf.py', type: 'file', comment: 'Item-based CF model' }, { name: 'matrix_factorization.py', type: 'file', comment: 'Implicit ALS wrapper' }, { name: 'ncf.py', type: 'file', comment: 'NCF model & wrapper' }, { name: 'popularity.py', type: 'file', comment: 'Popularity baseline model' }, ]}, { name: 'pipelines/', type: 'folder', comment: 'End-to-end execution scripts', children: [ { name: 'evaluate.py', type: 'file', comment: 'Evaluate a saved model' }, { name: 'run_preprocessing.py', type: 'file', comment: 'Execute full preprocessing' }, { name: 'setup_database.py', type: 'file', comment: 'Create/drop DB tables (Optional)' }, { name: 'train.py', type: 'file', comment: 'Train a specified model' }, ]}, ]},
        { name: 'tests/', type: 'folder', comment: 'Pytest unit/integration tests', children: [ ] }, ]};
const projectTreeData = addPaths(rawTreeData);
const GITHUB_BASE_URL = "https://github.com/mohitbhimrajka/recsys_final/blob/main";

// Wrapper Component for Tree View (Now defaults to expanded)
const InteractiveDirectoryTree: React.FC<{ structure: any }> = ({ structure }) => {
    const [searchTerm, setSearchTerm] = useState('');
    // *** MODIFICATION: Default state for allExpanded is now true ***
    const [allExpanded, setAllExpanded] = useState(true);

    const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setSearchTerm(event.target.value);
        // Expand automatically only if searching *and* currently collapsed
        if (event.target.value !== '' && !allExpanded) {
            setAllExpanded(true);
        }
    };

    const toggleAll = () => setAllExpanded(!allExpanded);

    return (
        <div className="bg-surface p-4 md:p-6 rounded-lg border border-border-color shadow-inner">
            {/* Controls: Search and Expand/Collapse */}
            <div className="flex flex-col sm:flex-row gap-3 mb-4">
                <input type="text" placeholder="Search files/folders/comments..." value={searchTerm} onChange={handleSearchChange} className="flex-grow px-3 py-1.5 bg-background border border-border-color rounded-md text-sm text-text-primary placeholder-text-muted focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary" />
                <button onClick={toggleAll} className="px-3 py-1.5 text-xs font-medium bg-background border border-border-color rounded-md text-text-secondary hover:border-primary hover:text-primary transition-colors whitespace-nowrap">
                    {allExpanded ? 'Collapse All' : 'Expand All'}
                </button>
            </div>

            {/* Tree */}
            <div className="max-h-[70vh] overflow-y-auto pr-2 text-sm"> {/* Slightly increased max-h */}
                <TreeNodeComponent node={structure} level={0} githubBaseUrl={GITHUB_BASE_URL} searchTerm={searchTerm} allExpanded={allExpanded} filterVisible={true} />
            </div>
        </div>
    );
};

// --- Main CodeExplorerPage Component ---
const CodeExplorerPage: React.FC = () => {
    return (
        <div className="py-16 md:py-24">
            <motion.h1 className="text-4xl md:text-5xl font-bold text-center mb-16 md:mb-20 text-text-primary" initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
                Project Structure & Code Highlights
            </motion.h1>

            {/* Interactive Directory Structure */}
            <motion.div className="mb-16 md:mb-20" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5, delay: 0.1 }}>
                <h2 className="text-2xl md:text-3xl lg:text-4xl font-semibold text-text-primary mb-4">Directory Overview</h2>
                 {/* *** ADDED: Explanatory Text *** */}
                 <p className="text-text-muted text-sm mb-6 flex items-center gap-2">
                     <FiInfo size={16} className="flex-shrink-0"/>
                     This is an interactive view of the project structure. Click folders <FiMinusSquare size={12}/> / <FiPlusSquare size={12}/> to toggle, hover for full paths, search, or click <FiExternalLink size={12}/> to view on GitHub.
                 </p>
                <InteractiveDirectoryTree structure={projectTreeData} />
            </motion.div>

            {/* Key Areas Highlight (Unchanged) */}
            <div>
                <h2 className="text-2xl md:text-3xl lg:text-4xl font-semibold text-text-primary mb-10">Code Highlights</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8 items-stretch">
                    <HighlightItem icon={<FiFilter />} title="Data Preprocessing" path="src/data/preprocess.py" githubUrl={`${GITHUB_BASE_URL}/src/data/preprocess.py`} description="Cleans raw OULAD data, filters interactions based on registration periods and activity thresholds (e.g., min 5 interactions/users), aggregates VLE clicks into an <code>implicit_feedback</code> score using <code>log1p</code>, and generates final user/item feature Parquet files." codeSnippet={preprocessSnippet} language="python" />
                    <HighlightItem icon={<FiCpu />} title="ItemCF Model Logic" path="src/models/item_cf.py" githubUrl={`${GITHUB_BASE_URL}/src/models/item_cf.py`} description="Item-Based Collaborative Filtering logic. Computes a cosine similarity matrix between courses based on co-interaction patterns. Predictions sum similarities weighted by the user's past interaction scores. This model performed best in offline tests." codeSnippet={itemcfSnippet} language="python" />
                    <HighlightItem icon={<FiZap />} title="API Service Logic" path="api/app/services.py" githubUrl={`${GITHUB_BASE_URL}/api/app/services.py`} description="Core logic for the FastAPI backend. Includes functions to get candidate items, predict scores using loaded models (like <code>model.predict(user_id, items)</code>), calculate weighted ensemble scores, and format results using Pydantic schemas." codeSnippet={apiServiceSnippet} language="python" />
                    <HighlightItem icon={<FiBarChart />} title="Evaluation Framework" path="src/evaluation/evaluator.py" githubUrl={`${GITHUB_BASE_URL}/src/evaluation/evaluator.py`} description="Implements evaluation using a time-based split. The <code>RecEvaluator</code> class calculates ranking metrics (Precision@k, Recall@k, NDCG@k) comparing model predictions against ground truth. Supports negative sampling for efficiency." codeSnippet={evaluatorSnippet} language="python" />
                    <HighlightItem icon={<FiLayout />} title="Frontend UI (React)" path="frontend/src/" githubUrl={`${GITHUB_BASE_URL}/frontend/src`} description="This interactive interface! Built using React, TypeScript, Vite, and Tailwind CSS. Key components include <code>UserSelector</code> (async search), <code>RecommendationCard</code>, and pages like <code>DemoPage</code> (with tabs), managing state and fetching API data." />
                    <HighlightItem icon={<FiBox />} title="Model Abstraction" path="src/models/base.py" githubUrl={`${GITHUB_BASE_URL}/src/models/base.py`} description="Defines the <code>BaseRecommender</code> abstract class. All models (Popularity, ItemCF, ALS, NCF, Hybrid) inherit from it, ensuring a consistent interface with <code>fit()</code>, <code>predict()</code>, <code>get_known_users()</code>, and <code>get_known_items()</code> methods, crucial for standardized evaluation and API usage." />
                </div>
            </div>
        </div>
    );
};

export default CodeExplorerPage;