import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiChevronDown, FiChevronRight, FiCode, FiLayout, FiZap, FiFilter, FiCpu, FiBarChart, FiHash } from 'react-icons/fi';
import { optimizedVariants } from '../../utils/animationUtils';
import CodeCard from './CodeCard';
import { CodeCategory, CodeItem, CategoryInfo, CategoryDisplayInfo } from '../../types/codeExplorer';

// Define the category display info
const categoryDisplayInfo: CategoryDisplayInfo = {
  all: { name: 'All Categories', icon: <FiHash /> },
  python: { name: 'Python', icon: <FiCode /> },
  typescript: { name: 'TypeScript', icon: <FiCode /> },
  frontend: { name: 'Frontend', icon: <FiLayout /> },
  backend: { name: 'Backend', icon: <FiZap /> },
  data: { name: 'Data Processing', icon: <FiFilter /> },
  models: { name: 'Models', icon: <FiCpu /> },
  evaluation: { name: 'Evaluation', icon: <FiBarChart /> }
};

interface CategorySectionProps {
  category: CodeCategory;
  items: CodeItem[];
  onItemClick: (itemId: string) => void;
}

const CategorySection: React.FC<CategorySectionProps> = ({ 
  category, 
  items, 
  onItemClick 
}) => {
  const [isCategoryExpanded, setIsCategoryExpanded] = useState(true);
  
  // Don't render empty categories
  if (items.length === 0) return null;
  
  const { name, icon } = categoryDisplayInfo[category];
  
  return (
    <div className="mb-12">
      <div className="flex items-center justify-between mb-4 group">
        <button 
          onClick={() => setIsCategoryExpanded(!isCategoryExpanded)}
          className="flex items-center gap-2 text-lg font-semibold text-text-primary hover:text-primary transition-colors"
          aria-expanded={isCategoryExpanded}
        >
          <span className="text-primary">{icon}</span>
          <h3>{name}</h3>
          <span className="transform transition-transform">
            {isCategoryExpanded ? 
              <FiChevronDown className="text-text-muted group-hover:text-primary" /> : 
              <FiChevronRight className="text-text-muted group-hover:text-primary" />
            }
          </span>
          <span className="text-xs text-text-muted ml-2 font-normal">
            ({items.length} {items.length === 1 ? 'snippet' : 'snippets'})
          </span>
        </button>
      </div>
      
      <AnimatePresence initial={false}>
        {isCategoryExpanded && (
          <motion.div 
            variants={optimizedVariants.expand}
            initial="collapsed"
            animate="expanded"
            exit="collapsed"
            className="overflow-hidden"
            style={{ willChange: 'height, opacity', transform: 'translateZ(0)' }}
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-stretch">
              {items.map(item => (
                <CodeCard 
                  key={item.id} 
                  item={item} 
                  onClick={() => onItemClick(item.id)} 
                />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default CategorySection; 