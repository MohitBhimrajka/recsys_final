// frontend/src/pages/CodeExplorerPage.tsx
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { FiFolder, FiFileText, FiCpu, FiDatabase, FiLayout, FiBarChart, FiZap, FiSettings, FiMinusSquare, FiPlusSquare } from 'react-icons/fi';

// --- Interactive Directory Tree Component ---
interface TreeNode {
    name: string;
    type: 'folder' | 'file';
    comment?: string;
    children?: TreeNode[];
}

const projectTreeData: TreeNode = {
    name: 'mohitbhimrajka-recsys_final/', type: 'folder', children: [
        { name: 'README.md', type: 'file', comment: 'Project overview & setup' },
        { name: 'requirements.txt', type: 'file', comment: 'Core Python dependencies' },
        { name: '.env.example', type: 'file', comment: 'Database credential template' },
        { name: 'api/', type: 'folder', comment: 'FastAPI Backend Server', children: [
            { name: 'requirements.txt', type: 'file', comment: 'API specific dependencies' },
            { name: 'app/', type: 'folder', children: [
                { name: 'main.py', type: 'file', comment: 'App setup, CORS, startup event' },
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
                 /* ... other src files */
            ]}
        ]},
        { name: 'notebooks/', type: 'folder', comment: 'Jupyter Dev Notebooks', children: [
            { name: '01_eda.ipynb', type: 'file' },
            { name: '02_preprocessing_feature_eng.ipynb', type: 'file'},
            { name: '03_baseline_models.ipynb', type: 'file'},
            /* ... other notebooks */
        ]},
        { name: 'reports/', type: 'folder', comment: 'Markdown Reports' },
        { name: 'src/', type: 'folder', comment: 'Core Python Library', children: [
            { name: 'config.py', type: 'file', comment: 'Paths, params, DB URI' },
            { name: 'data/', type: 'folder', children: [
                 { name: 'preprocess.py', type: 'file', comment: 'Data cleaning/feature eng.' },
                 { name: 'dataset.py', type: 'file', comment: 'PyTorch Dataset classes' },
            ]},
            { name: 'database/', type: 'folder', comment: 'DB schema & loading' },
            { name: 'evaluation/', type: 'folder', children: [ { name: 'evaluator.py', type: 'file', comment: 'Calculates metrics' } ]},
            { name: 'models/', type: 'folder', children: [
                 { name: 'base.py', type: 'file', comment: 'Abstract model class' },
                 { name: 'item_cf.py', type: 'file', comment: 'ItemCF implementation' },
                 /* ... other models */
            ]},
            { name: 'pipelines/', type: 'folder', children: [
                 { name: 'run_preprocessing.py', type: 'file' },
                 { name: 'train.py', type: 'file' },
                 { name: 'evaluate.py', type: 'file' },
            ]},
        ]},
        { name: 'tests/', type: 'folder', comment: 'Pytest Tests' },
    ]
};

const TreeNodeComponent: React.FC<{ node: TreeNode; level: number }> = ({ node, level }) => {
  const [isOpen, setIsOpen] = useState(level < 1);
  const isFolder = node.type === 'folder';
  const toggleOpen = () => { if (isFolder) setIsOpen(!isOpen); };
  const iconColor = isFolder ? (isOpen ? 'text-primary' : 'text-text-muted') : 'text-text-muted';
  const textColor = isFolder ? 'text-text-primary font-medium' : 'text-text-secondary';

  return (
    <div style={{ paddingLeft: `${level * 20}px` }} className="text-sm font-mono select-none">
      <div
        className={`flex items-center py-1 group hover:bg-border-color/20 rounded ${isFolder ? 'cursor-pointer' : ''}`}
        onClick={toggleOpen} role={isFolder ? 'button' : undefined} tabIndex={isFolder ? 0 : -1}
        onKeyDown={(e) => { if (isFolder && (e.key === 'Enter' || e.key === ' ')) toggleOpen(); }}
      >
        <span className="w-[14px] text-center mr-2 flex-shrink-0 text-text-muted">
          {isFolder && (isOpen ? <FiMinusSquare size={14} /> : <FiPlusSquare size={14} />)}
        </span>
        {isFolder ? <FiFolder size={16} className={`mr-2 flex-shrink-0 ${iconColor}`} /> : <FiFileText size={16} className={`mr-2 flex-shrink-0 ${iconColor}`} />}
        <span className={`${textColor}`}>{node.name}</span>
        {node.comment && <span className="ml-3 text-xs text-text-muted italic opacity-0 group-hover:opacity-100 transition-opacity">// {node.comment}</span>}
      </div>
      {isFolder && isOpen && node.children && (
        <motion.div
            initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }} className="border-l border-dashed border-border-color ml-[7px] overflow-hidden"
        >
          {node.children.map((child, index) => ( <TreeNodeComponent key={`${child.name}-${index}`} node={child} level={level + 1} /> ))}
        </motion.div>
      )}
    </div>
  );
};

const InteractiveDirectoryTree: React.FC<{ structure: TreeNode }> = ({ structure }) => (
  <div className="bg-surface p-4 md:p-6 rounded-lg border border-border-color shadow-inner max-h-[600px] overflow-y-auto">
      <TreeNodeComponent node={structure} level={0} />
  </div>
);

// HighlightItem component
const HighlightItem: React.FC<{ icon: React.ReactNode; title: string; description: string; path?: string }> = ({ icon, title, description, path }) => (
    <motion.div
      className="bg-surface p-6 rounded-xl border border-border-color shadow-lg h-full flex flex-col transform transition duration-300 hover:border-primary/50 hover:-translate-y-1.5 hover:shadow-primary/10"
      initial={{ opacity: 0, y: 15 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.2 }} // Adjusted amount
      transition={{ duration: 0.4, ease: 'easeOut' }}
    >
      <div className="flex items-center mb-4">
        <span className="text-primary mr-4 text-2xl flex-shrink-0">{icon}</span>
        <h4 className="text-lg font-semibold text-text-primary flex-1 leading-snug">{title}</h4>
      </div>
       <div className="flex-grow">
           {path && ( <p className="text-xs font-mono bg-background inline-block px-2 py-1 rounded mb-4 text-primary/80 border border-border-color break-all">{path}</p> )}
           <p className="text-sm text-text-muted">{description}</p>
       </div>
    </motion.div>
  );

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
             icon={<FiDatabase />} title="Data Processing Pipeline" path="src/data/preprocess.py"
             description="Cleans raw OULAD CSVs, handles missing values, filters interactions based on activity and registration, aggregates engagement metrics, and generates final user/item feature Parquet files." />
           <HighlightItem
             icon={<FiCpu />} title="Recommendation Models" path="src/models/"
             description="Contains Python implementations of various recommendation algorithms (Popularity, ItemCF, ALS Wrapper, NCF, Hybrid). ItemCF is used in the live demo." />
           <HighlightItem
             icon={<FiSettings />} title="ItemCF Model (Demo)" path="src/models/item_cf.py"
             description="The specific Item-Based Collaborative Filtering logic used in the demo. Calculates item similarity and predicts scores based on a user's past interactions." />
           <HighlightItem
             icon={<FiZap />} title="Backend API (FastAPI)" path="api/app/"
             description="Serves the pre-trained ItemCF model via HTTP endpoints. Handles requests for user search, random user selection, and recommendation generation." />
           <HighlightItem
             icon={<FiLayout />} title="Frontend UI (React)" path="frontend/src/"
             description="This interactive interface! Built with React, TypeScript, Vite, and Tailwind CSS. Manages state, calls the API, and renders the user experience." />
           <HighlightItem
             icon={<FiBarChart />} title="Evaluation Framework" path="src/evaluation/evaluator.py"
             description="Implements the evaluation protocol using a time-based split and calculates standard ranking metrics (Precision@k, Recall@k, NDCG@k) to assess model performance." />
        </div>
      </div>
    </div>
  );
};

export default CodeExplorerPage;