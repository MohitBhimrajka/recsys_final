// frontend/src/pages/CodeExplorerPage.tsx
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { FiCode, FiDatabase, FiLayout, FiBarChart, FiZap, FiSettings, FiCpu } from 'react-icons/fi';
import TreeNodeComponent from '../components/TreeNodeComponent'; // Import extracted component
import HighlightItem from '../components/HighlightItem'; // Import extracted component

// --- Project Tree Data (with paths) ---
// Function to recursively add paths
const addPaths = (node: any, currentPath: string = ''): any => {
    const nodePath = currentPath ? `${currentPath}/${node.name}` : node.name;
    const newNode = { ...node, path: nodePath };
    if (node.children) {
        newNode.children = node.children.map((child: any) => addPaths(child, nodePath));
    }
    return newNode;
};

const rawTreeData = {
    name: 'recsys_final/', type: 'folder', children: [
        { name: 'README.md', type: 'file', comment: 'Project overview & setup' },
        { name: 'requirements.txt', type: 'file', comment: 'Core Python dependencies' },
        { name: '.env.example', type: 'file', comment: 'Database credential template' },
        { name: 'api/', type: 'folder', comment: 'FastAPI Backend Server', children: [
            { name: 'requirements.txt', type: 'file', comment: 'API specific dependencies' },
            { name: 'app/', type: 'folder', children: [
                { name: 'main.py', type: 'file', comment: 'App setup, CORS, startup' },
                { name: 'model_loader.py', type: 'file', comment: 'Loads trained ItemCF model' },
                { name: 'schemas.py', type: 'file', comment: 'Pydantic API models' },
                { name: 'services.py', type: 'file', comment: 'Core recommendation logic' },
                { name: 'routers/', type: 'folder', children: [ { name: 'recommendations.py', type: 'file', comment: 'API endpoints' } ]}
            ]}
        ]},
        { name: 'frontend/', type: 'folder', comment: 'React/Vite Frontend', children: [
            { name: 'vite.config.ts', type: 'file' },
            { name: 'tailwind.config.js', type: 'file' },
            { name: 'index.html', type: 'file' },
            { name: 'src/', type: 'folder', children: [
                 { name: 'components/', type: 'folder', comment: 'UI elements' },
                 { name: 'pages/', type: 'folder', comment: 'Page views' },
                 { name: 'services/', type: 'folder', comment: 'API calls' },
                 { name: 'App.tsx', type: 'file' }, { name: 'main.tsx', type: 'file' }
            ]}
        ]},
        { name: 'notebooks/', type: 'folder', comment: 'Jupyter Dev Notebooks', children: [
            { name: '01_eda.ipynb', type: 'file' },
            { name: '02_preprocessing_feature_eng.ipynb', type: 'file'},
            { name: '03_baseline_models.ipynb', type: 'file'},
            { name: '04_ncf_dev.ipynb', type: 'file' },
            { name: '05_content_hybrid_dev.ipynb', type: 'file' },
        ]},
        { name: 'reports/', type: 'folder', comment: 'Markdown Reports' },
        { name: 'src/', type: 'folder', comment: 'Core Python Library', children: [
            { name: 'config.py', type: 'file', comment: 'Paths, params, DB URI' },
            { name: 'data/', type: 'folder', children: [
                 { name: 'preprocess.py', type: 'file', comment: 'Data cleaning/feature eng.' },
                 { name: 'dataset.py', type: 'file', comment: 'PyTorch Dataset classes' },
                 { name: 'load_raw.py', type: 'file' }, { name: 'utils.py', type: 'file' },
            ]},
            { name: 'database/', type: 'folder', comment: 'DB schema & loading', children: [
                 { name: 'schema.py', type: 'file' }, { name: 'load_to_db.py', type: 'file' }, { name: 'db_utils.py', type: 'file' },
            ]},
            { name: 'evaluation/', type: 'folder', children: [ { name: 'evaluator.py', type: 'file' }, { name: 'metrics.py', type: 'file' } ]},
            { name: 'models/', type: 'folder', children: [
                 { name: 'base.py', type: 'file', comment: 'Abstract model class' },
                 { name: 'item_cf.py', type: 'file', comment: 'ItemCF implementation' },
                 { name: 'ncf.py', type: 'file' }, { name: 'hybrid.py', type: 'file' }, { name: 'popularity.py', type: 'file' },
                 { name: 'matrix_factorization.py', type: 'file' }, { name: 'content_encoder.py', type: 'file' },
            ]},
            { name: 'pipelines/', type: 'folder', children: [
                 { name: 'run_preprocessing.py', type: 'file' },
                 { name: 'train.py', type: 'file' },
                 { name: 'evaluate.py', type: 'file' }, { name: 'setup_database.py', type: 'file' },
            ]},
        ]},
        { name: 'tests/', type: 'folder', comment: 'Pytest Tests' },
        { name: 'pytest.ini', type: 'file' },
        { name: '.gitignore', type: 'file' },
    ]
};

// Add paths to the tree structure
const projectTreeData = addPaths(rawTreeData);

// GitHub base URL (replace with your actual repo URL)
const GITHUB_BASE_URL = "https://github.com/mohitbhimrajka/recsys_final/blob/main"; // Adjust branch if needed

// Interactive Directory Tree Component (Wrapper)
const InteractiveDirectoryTree: React.FC<{ structure: any }> = ({ structure }) => {
    const [searchTerm, setSearchTerm] = useState('');
    const [allExpanded, setAllExpanded] = useState(false);

    const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setSearchTerm(event.target.value);
        // Optionally expand all on search? Or leave it manual.
        // if (event.target.value !== '' && !allExpanded) setAllExpanded(true);
    };

    const toggleAll = () => setAllExpanded(!allExpanded);

    return (
        <div className="bg-surface p-4 md:p-6 rounded-lg border border-border-color shadow-inner">
            {/* Controls: Search and Expand/Collapse */}
            <div className="flex flex-col sm:flex-row gap-3 mb-4">
                <input
                    type="text"
                    placeholder="Search files/folders..."
                    value={searchTerm}
                    onChange={handleSearchChange}
                    className="flex-grow px-3 py-1.5 bg-background border border-border-color rounded-md text-sm text-text-primary placeholder-text-muted focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary"
                />
                <button
                    onClick={toggleAll}
                    className="px-3 py-1.5 text-xs font-medium bg-background border border-border-color rounded-md text-text-secondary hover:border-primary hover:text-primary transition-colors"
                >
                    {allExpanded ? 'Collapse All' : 'Expand All'}
                </button>
            </div>

            {/* Tree */}
            <div className="max-h-[600px] overflow-y-auto pr-2">
                <TreeNodeComponent
                     node={structure}
                     level={0}
                     githubBaseUrl={GITHUB_BASE_URL}
                     searchTerm={searchTerm}
                     allExpanded={allExpanded}
                     filterVisible={true} // Root is always visible
                 />
            </div>
        </div>
    );
};


// --- Code Snippets ---
const preprocessSnippet = `
def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    # ... (previous code) ...
    daily_interactions = df.groupby(['id_student', 'presentation_id', 'date'])['sum_click'].sum().reset_index()
    user_item_interactions = daily_interactions.groupby(['id_student', 'presentation_id']).agg(
        total_clicks=('sum_click', 'sum'),
        interaction_days=('date', 'nunique'),
        first_interaction_date=('date', 'min'),
        last_interaction_date=('date', 'max')
    ).reset_index()
    # Calculate implicit feedback score (log transformation)
    user_item_interactions['implicit_feedback'] = np.log1p(user_item_interactions['total_clicks'])
    return user_item_interactions
`;

const itemcfSnippet = `
def predict(self, user_id: Any, item_ids: List[Any]) -> List[float]:
    user_idx = self.user_id_to_idx.get(user_id)
    if user_idx is None: return [0.0] * len(item_ids)
    # ... (check model fitted) ...

    user_interactions_vector = self.interaction_matrix_sparse[user_idx]
    target_item_indices = [...] # Map item_ids to internal indices

    scores = [0.0] * len(item_ids)
    for i, target_idx in enumerate(target_item_indices):
        if target_idx is None: continue

        # Get pre-calculated similarities for the target item
        target_item_similarities = self.item_similarity_matrix[target_idx]

        # Dot product to get weighted sum of interactions
        score = user_interactions_vector.dot(target_item_similarities.T).toarray()[0, 0]
        scores[i] = float(score)
    return scores
`;

const apiServiceSnippet = `
def get_recommendations_for_user(user_id: int, k: int, model, ...) -> List[schemas.RecommendationItem]:
    if user_id not in model.get_known_users(): return []

    items_seen_by_user = train_map.get(user_id, set())
    candidate_items = list(all_items - items_seen_by_user)
    if not candidate_items: return []

    scores = model.predict(user_id, candidate_items) # Get scores from model

    item_score_pairs = list(zip(candidate_items, scores))
    item_score_pairs.sort(key=lambda x: x[1], reverse=True)
    top_k_pairs = item_score_pairs[:k]

    # Format results with details
    results = []
    for item_id, score in top_k_pairs:
        details = item_details.loc[item_id] # ... fetch details ...
        results.append(schemas.RecommendationItem(...))
    return results
`;

const CodeExplorerPage: React.FC = () => {
    return (
        <div className="container mx-auto px-4 py-16 md:py-24 max-w-6xl">
            <motion.h1
                className="text-4xl md:text-5xl font-bold text-center mb-16 md:mb-20 text-text-primary"
                initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
            >
                Project Structure & Code Highlights
            </motion.h1>

            {/* Interactive Directory Structure */}
            <motion.div
                className="mb-16 md:mb-20"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5, delay: 0.1 }}
            >
                <h2 className="text-2xl md:text-3xl lg:text-4xl font-semibold text-text-primary mb-8">Directory Overview</h2>
                <InteractiveDirectoryTree structure={projectTreeData} />
            </motion.div>

            {/* Key Areas Highlight */}
            <div>
                <h2 className="text-2xl md:text-3xl lg:text-4xl font-semibold text-text-primary mb-10">Code Highlights</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8 items-stretch">
                    <HighlightItem
                        icon={<FiDatabase />} title="Data Processing" path="src/data/preprocess.py"
                        githubUrl={`${GITHUB_BASE_URL}/src/data/preprocess.py`}
                        description="Cleans raw data, filters interactions based on registration and activity thresholds (e.g., <code>MIN_INTERACTIONS_PER_USER=5</code>), aggregates VLE clicks into an <code>implicit_feedback</code> score (using <code>log1p</code>), and generates final user/item feature Parquet files."
                        codeSnippet={preprocessSnippet}
                        language="python"
                    />
                    <HighlightItem
                        icon={<FiCpu />} title="Recommendation Models" path="src/models/"
                        githubUrl={`${GITHUB_BASE_URL}/src/models`}
                        description="Contains implementations for various algorithms: <code>Popularity</code>, <code>ItemCF</code>, <code>ALS</code> (via implicit lib), <code>NCF</code>, and <code>HybridNCF</code>. All inherit from a common <code>BaseRecommender</code> ensuring a consistent <code>fit/predict</code> interface."
                    />
                    <HighlightItem
                        icon={<FiSettings />} title="ItemCF Model (Demo)" path="src/models/item_cf.py"
                        githubUrl={`${GITHUB_BASE_URL}/src/models/item_cf.py`}
                        description="The Item-Based Collaborative Filtering logic used in the live demo. It computes a cosine similarity matrix between course presentations based on co-interactions. Predictions are made by summing similarities weighted by the user's past interaction scores."
                        codeSnippet={itemcfSnippet}
                        language="python"
                    />
                    <HighlightItem
                        icon={<FiZap />} title="Backend API Service" path="api/app/services.py"
                        githubUrl={`${GITHUB_BASE_URL}/api/app/services.py`}
                        description="Core logic for the FastAPI backend. The <code>get_recommendations_for_user</code> function takes a user ID, retrieves seen items, gets predictions from the loaded model for unseen candidates, and formats the top-K results using Pydantic schemas."
                        codeSnippet={apiServiceSnippet}
                        language="python"
                    />
                    <HighlightItem
                        icon={<FiLayout />} title="Frontend UI (React)" path="frontend/src/"
                        githubUrl={`${GITHUB_BASE_URL}/frontend/src`}
                        description="This interactive interface! Built using React, TypeScript, Vite, and Tailwind CSS. Key components include <code>UserSelector</code> (async search), <code>RecommendationCard</code>, and pages like <code>DemoPage</code> which manage state, fetch data via services, and render the UI."
                    />
                    <HighlightItem
                        icon={<FiBarChart />} title="Evaluation Framework" path="src/evaluation/evaluator.py"
                        githubUrl={`${GITHUB_BASE_URL}/src/evaluation/evaluator.py`}
                        description="Implements the evaluation protocol using a time-based split. The <code>RecEvaluator</code> class calculates standard ranking metrics (Precision@k, Recall@k, NDCG@k) comparing model predictions against the test set ground truth. Supports negative sampling for efficiency."
                    />
                </div>
            </div>
        </div>
    );
};

export default CodeExplorerPage;