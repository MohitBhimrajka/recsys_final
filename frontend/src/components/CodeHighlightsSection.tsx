import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiFilter, FiCpu, FiZap, FiBarChart, FiLayout, FiBox, FiCode, FiList, FiGrid, FiMaximize2, FiX, FiSearch, FiChevronDown, FiChevronRight, FiHash, FiArrowUp, FiStar } from 'react-icons/fi';
import HighlightItem from './HighlightItem';
import { optimizedVariants, smoothTransition, hardwareAcceleration } from '../utils/animationUtils';
import CategorySection from './code-explorer/CategorySection';
import CodeCard from './code-explorer/CodeCard';
import { CodeCategory, CodeItem, CategoryDisplayInfo } from '../types/codeExplorer';

// Display names and icons for each category
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

interface CodeHighlightsSectionProps {
  codeItems: CodeItem[];
  githubBaseUrl: string;
}

const CodeHighlightsSection: React.FC<CodeHighlightsSectionProps> = ({ codeItems, githubBaseUrl }) => {
  const [activeFilter, setActiveFilter] = useState<CodeCategory>('all');
  const [expandedItem, setExpandedItem] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'grid' | 'list' | 'categorized'>('categorized');
  const [searchQuery, setSearchQuery] = useState('');
  const [scrollPosition, setScrollPosition] = useState(0);
  const [showScrollTop, setShowScrollTop] = useState(false);

  // Track scroll position to show/hide scroll-to-top button
  useEffect(() => {
    const handleScroll = () => {
      const position = window.pageYOffset;
      setScrollPosition(position);
      setShowScrollTop(position > 500);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Scroll to top function
  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  };

  // Filter code items based on active filter and search query
  const filteredItems = codeItems.filter(item => {
    const matchesFilter = activeFilter === 'all' || item.category.includes(activeFilter);
    const matchesSearch = searchQuery === '' || 
      item.title.toLowerCase().includes(searchQuery.toLowerCase()) || 
      item.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.path.toLowerCase().includes(searchQuery.toLowerCase());
    
    return matchesFilter && matchesSearch;
  });

  // Organize items by category (for categorized view)
  const itemsByCategory = React.useMemo(() => {
    const categories: CodeCategory[] = ['python', 'typescript', 'frontend', 'backend', 'data', 'models', 'evaluation'];
    const result: Record<CodeCategory, CodeItem[]> = {
      all: [],
      python: [],
      typescript: [],
      frontend: [],
      backend: [],
      data: [],
      models: [],
      evaluation: []
    };

    // Only process if we're in categorized view or showing all items
    if (viewMode === 'categorized' || activeFilter === 'all') {
      filteredItems.forEach(item => {
        // If we have a specific filter, only add to that category
        if (activeFilter !== 'all') {
          if (item.category.includes(activeFilter)) {
            result[activeFilter].push(item);
          }
        } else {
          // With 'all' filter, add to each relevant category
          item.category.forEach(cat => {
            if (cat !== 'all') {
              result[cat].push(item);
            }
          });
          // Also add to 'all' for completeness
          result.all.push(item);
        }
      });
    }

    return result;
  }, [filteredItems, activeFilter, viewMode]);

  // Find featured item for categorized view
  const featuredItem = React.useMemo(() => {
    return filteredItems.find(item => item.isFeatured);
  }, [filteredItems]);

  // Full-screen modal for code viewing
  const FullScreenModal = ({ item }: { item: CodeItem }) => (
    <motion.div 
      className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center p-4 sm:p-8"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      style={hardwareAcceleration}
    >
      <motion.div 
        className="bg-surface/90 rounded-xl border border-border-color/70 shadow-2xl w-full max-w-6xl max-h-[90vh] flex flex-col overflow-hidden"
        initial={{ scale: 0.9, y: 20 }}
        animate={{ scale: 1, y: 0 }}
        exit={{ scale: 0.9, y: 20 }}
        transition={smoothTransition}
        style={hardwareAcceleration}
      >
        <div className="flex items-center justify-between p-4 border-b border-border-color/70">
          <div className="flex items-center gap-3">
            <span className="text-primary text-2xl">{item.icon}</span>
            <h3 className="text-xl font-semibold text-text-primary">{item.title}</h3>
          </div>
          <button 
            onClick={() => setExpandedItem(null)}
            className="p-2 rounded-full hover:bg-black/20 text-text-muted hover:text-text-primary transition-colors"
          >
            <FiX size={24} />
          </button>
        </div>
        <div className="p-4 overflow-y-auto flex-grow">
          <p className="text-sm font-mono text-primary/80 bg-background inline-block px-3 py-2 rounded mb-4 border border-border-color">
            {item.path}
          </p>
          <div className="prose prose-sm prose-invert max-w-none mb-6 text-text-muted">
            <p dangerouslySetInnerHTML={{ __html: item.description }}></p>
          </div>
          {item.codeSnippet && (
            <div className="bg-black/80 rounded-lg border border-border-color/50 p-5 text-sm shadow-inner">
              <HighlightItem
                icon={<></>} // Empty icon as we already show it in the header
                title=""
                description=""
                path=""
                githubUrl={item.githubUrl}
                codeSnippet={item.codeSnippet}
                language={item.language}
                expandedView={true}
              />
            </div>
          )}
        </div>
        <div className="p-4 border-t border-border-color/70 flex justify-end">
          <a 
            href={item.githubUrl} 
            target="_blank" 
            rel="noopener noreferrer"
            className="px-4 py-2 rounded-md bg-primary/10 text-primary hover:bg-primary/20 transition-colors text-sm font-medium"
          >
            View on GitHub
          </a>
        </div>
      </motion.div>
    </motion.div>
  );

  // Enhanced section for featured example
  const FeaturedCodeExample = ({ item }: { item: CodeItem }) => {
    if (!item) return null;
    
    return (
      <motion.div 
        className="bg-surface border-l-4 border-primary rounded-lg shadow-sm overflow-hidden mb-10"
        initial="hidden"
        animate="visible"
        variants={optimizedVariants}
        style={hardwareAcceleration}
      >
        <div className="p-4 bg-gradient-to-r from-primary/10 to-transparent">
          <div className="flex items-center gap-2 mb-2">
            <FiStar className="text-primary" size={20} />
            <h3 className="text-lg font-medium text-primary">Featured Example</h3>
          </div>
          
          <div className="flex flex-wrap gap-2 mt-2 mb-3">
            {item.category.map(cat => (
              <span key={cat} className="text-xs bg-primary/20 text-primary px-2 py-1 rounded-full">
                {categoryDisplayInfo[cat as CodeCategory].name}
              </span>
            ))}
          </div>
        </div>
        
        <div 
          className="p-4 cursor-pointer transition-all hover:bg-background/50" 
          onClick={() => setExpandedItem(item.id)}
        >
          <div className="flex items-center gap-3 mb-3">
            <span className="text-primary text-xl">{item.icon}</span>
            <h4 className="font-medium text-text-primary text-lg">{item.title}</h4>
          </div>
          
          <p className="text-sm font-mono text-primary/80 bg-background inline-block px-2 py-1 rounded mb-3">
            {item.path}
          </p>
          
          <p className="text-text-muted mb-4">
            {item.description}
          </p>
          
          <div className="flex justify-end">
            <button 
              onClick={(e) => {
                e.stopPropagation();
                setExpandedItem(item.id);
              }} 
              className="px-4 py-2 bg-primary/10 hover:bg-primary/20 text-primary rounded-md transition-colors text-sm font-medium"
            >
              View Details
            </button>
          </div>
        </div>
      </motion.div>
    );
  };

  return (
    <div className="space-y-8">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <h2 className="text-2xl md:text-3xl lg:text-4xl font-semibold text-text-primary">Code Highlights</h2>
        
        <div className="flex flex-col sm:flex-row gap-3">
          {/* Search Bar */}
          <div className="relative">
            <FiSearch className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
            <input
              type="text"
              placeholder="Search code snippets..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 pr-4 py-2 bg-background border border-border-color rounded-md text-sm w-full sm:w-auto min-w-[200px] focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary"
            />
          </div>
          
          {/* View Mode Toggle */}
          <div className="flex bg-background border border-border-color rounded-md overflow-hidden">
            <button
              className={`p-2 flex items-center ${viewMode === 'grid' ? 'bg-primary/10 text-primary' : 'text-text-muted'}`}
              onClick={() => setViewMode('grid')}
              title="Grid View"
            >
              <FiGrid size={18} />
            </button>
            <button
              className={`p-2 flex items-center ${viewMode === 'list' ? 'bg-primary/10 text-primary' : 'text-text-muted'}`}
              onClick={() => setViewMode('list')}
              title="List View"
            >
              <FiList size={18} />
            </button>
            <button
              className={`p-2 flex items-center ${viewMode === 'categorized' ? 'bg-primary/10 text-primary' : 'text-text-muted'}`}
              onClick={() => setViewMode('categorized')}
              title="Categorized View"
            >
              <FiFilter size={18} />
            </button>
          </div>
        </div>
      </div>

      <p className="text-text-muted text-sm max-w-2xl">
        Explore key code snippets from the project's architecture. Filter by category, switch views, and click on any card to see a detailed view.
      </p>

      {/* Filter Tabs */}
      <div className="flex flex-wrap gap-2 mb-8">
        <FilterButton
          active={activeFilter === 'all'}
          onClick={() => setActiveFilter('all')}
          icon={<FiHash />}
          label="All"
        />
        <FilterButton
          active={activeFilter === 'python'}
          onClick={() => setActiveFilter('python')}
          icon={<FiCode />}
          label="Python"
        />
        <FilterButton
          active={activeFilter === 'frontend'}
          onClick={() => setActiveFilter('frontend')}
          icon={<FiLayout />}
          label="Frontend"
        />
        <FilterButton
          active={activeFilter === 'backend'}
          onClick={() => setActiveFilter('backend')}
          icon={<FiZap />}
          label="Backend"
        />
        <FilterButton
          active={activeFilter === 'data'}
          onClick={() => setActiveFilter('data')}
          icon={<FiFilter />}
          label="Data"
        />
        <FilterButton
          active={activeFilter === 'models'}
          onClick={() => setActiveFilter('models')}
          icon={<FiCpu />}
          label="Models"
        />
        <FilterButton
          active={activeFilter === 'evaluation'}
          onClick={() => setActiveFilter('evaluation')}
          icon={<FiBarChart />}
          label="Evaluation"
        />
      </div>

      {/* No Results Message */}
      {filteredItems.length === 0 && (
        <motion.div 
          initial={{ opacity: 0, y: 10 }} 
          animate={{ opacity: 1, y: 0 }} 
          className="text-center py-12 text-text-muted"
          style={hardwareAcceleration}
        >
          <p className="text-lg mb-2">No code snippets match your filters</p>
          <p className="text-sm">Try changing your search term or selecting a different category</p>
          <button 
            onClick={() => {setActiveFilter('all'); setSearchQuery('');}} 
            className="mt-4 px-4 py-2 bg-primary/10 text-primary rounded-md hover:bg-primary/20 transition-colors"
          >
            Reset Filters
          </button>
        </motion.div>
      )}

      {/* Featured item highlighted at the top (shown in all view modes when not searching) */}
      {searchQuery === '' && featuredItem && <FeaturedCodeExample item={featuredItem} />}

      {/* Main content area - show different layouts based on view mode */}
      {filteredItems.length > 0 && (
        <>
          {/* Categorized View */}
          {viewMode === 'categorized' && (
            <motion.div
              variants={optimizedVariants.container}
              initial="hidden"
              animate="visible"
              style={hardwareAcceleration}
            >
              {Object.keys(itemsByCategory)
                .filter(cat => cat !== 'all' && itemsByCategory[cat as CodeCategory].length > 0)
                .map((category) => (
                  <CategorySection 
                    key={category} 
                    category={category as CodeCategory} 
                    items={itemsByCategory[category as CodeCategory]}
                    onItemClick={(itemId) => setExpandedItem(itemId)}
                  />
                ))}
            </motion.div>
          )}

          {/* Grid View */}
          {viewMode === 'grid' && (
            <motion.div 
              variants={optimizedVariants.container}
              initial="hidden"
              animate="visible"
              className="grid grid-cols-1 lg:grid-cols-2 gap-6 md:gap-8 items-stretch"
              style={hardwareAcceleration}
            >
              {filteredItems.filter(item => !item.isFeatured || searchQuery !== '').map(item => (
                <CodeCard 
                  key={item.id} 
                  item={item} 
                  onClick={() => setExpandedItem(item.id)} 
                />
              ))}
            </motion.div>
          )}

          {/* List View */}
          {viewMode === 'list' && (
            <motion.div 
              variants={optimizedVariants.container}
              initial="hidden"
              animate="visible"
              className="space-y-3"
              style={hardwareAcceleration}
            >
              {filteredItems.filter(item => !item.isFeatured || searchQuery !== '').map(item => (
                <CodeCard 
                  key={item.id} 
                  item={item} 
                  onClick={() => setExpandedItem(item.id)} 
                  listView={true} 
                />
              ))}
            </motion.div>
          )}
        </>
      )}

      {/* Floating scroll-to-top button */}
      <AnimatePresence>
        {showScrollTop && (
          <motion.button
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            transition={smoothTransition}
            className="fixed bottom-8 right-8 p-3 bg-primary text-white rounded-full shadow-lg hover:bg-primary-dark focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 z-40"
            onClick={scrollToTop}
            aria-label="Scroll to top"
            style={hardwareAcceleration}
          >
            <FiArrowUp size={20} />
          </motion.button>
        )}
      </AnimatePresence>

      {/* Full-screen code view modal */}
      <AnimatePresence>
        {expandedItem && (
          <FullScreenModal 
            item={codeItems.find(item => item.id === expandedItem)!} 
          />
        )}
      </AnimatePresence>
    </div>
  );
};

// FilterButton Component
const FilterButton: React.FC<{
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}> = ({ active, onClick, icon, label }) => (
  <button
    className={`px-3 py-2 rounded-md flex items-center gap-2 text-sm transition-colors ${
      active
        ? 'bg-primary text-white'
        : 'bg-background text-text-muted hover:bg-border-color/30'
    }`}
    onClick={onClick}
  >
    {icon}
    <span>{label}</span>
  </button>
);

export type { CodeCategory }; // Export only the type, not the data
export default CodeHighlightsSection; 