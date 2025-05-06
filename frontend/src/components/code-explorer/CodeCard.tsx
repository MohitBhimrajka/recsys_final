import React from 'react';
import { motion } from 'framer-motion';
import { FiExternalLink } from 'react-icons/fi';
import { optimizedVariants } from '../../utils/animationUtils';
import { CodeItem } from '../../types/codeExplorer';

interface CodeCardProps {
  item: CodeItem;
  onClick: () => void;
  listView?: boolean;
}

const CodeCard: React.FC<CodeCardProps> = ({ item, onClick, listView = false }) => {
  const { icon, title, path, description, githubUrl } = item;
  
  // If in list view, use a more compact layout
  if (listView) {
    return (
      <motion.div
        className="flex items-center bg-surface border border-border-color/50 rounded-md hover:border-primary/30 transition-colors p-3 cursor-pointer"
        whileHover={{ y: -2, boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)' }}
        onClick={onClick}
        style={{ transform: 'translateZ(0)', willChange: 'transform' }}
      >
        <span className="text-primary text-lg mr-3">{icon}</span>
        <div className="flex-grow min-w-0">
          <h4 className="font-medium text-text-primary truncate">{title}</h4>
          <p className="text-xs text-text-muted">{path}</p>
        </div>
        <a
          href={githubUrl}
          target="_blank"
          rel="noopener noreferrer"
          onClick={(e) => e.stopPropagation()}
          className="text-text-muted hover:text-primary p-2 transition-colors"
          title="View on GitHub"
        >
          <FiExternalLink size={16} />
        </a>
      </motion.div>
    );
  }
  
  // Default grid card view
  return (
    <motion.div
      className="bg-surface border border-border-color/50 rounded-lg hover:border-primary/30 transition-colors cursor-pointer h-full flex flex-col"
      whileHover={{ y: -5 }}
      transition={{ type: "spring", stiffness: 300 }}
      onClick={onClick}
      variants={optimizedVariants}
      style={{ transform: 'translateZ(0)', willChange: 'transform' }}
    >
      <div className="p-4 flex-grow">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-primary text-xl">{icon}</span>
          <h4 className="font-medium text-text-primary">{title}</h4>
        </div>
        
        <p className="text-xs font-mono text-primary/80 bg-background inline-block px-2 py-1 rounded mb-3">
          {path}
        </p>
        
        <p className="text-sm text-text-muted line-clamp-3">
          {description}
        </p>
      </div>
      
      <div className="p-3 border-t border-border-color/30 flex justify-between items-center">
        <span className="text-xs text-text-muted">Click to expand</span>
        <a
          href={githubUrl}
          target="_blank"
          rel="noopener noreferrer"
          onClick={(e) => e.stopPropagation()}
          className="text-text-muted hover:text-primary p-1 transition-colors"
          title="View on GitHub"
        >
          <FiExternalLink size={14} />
        </a>
      </div>
    </motion.div>
  );
};

export default CodeCard; 