// frontend/src/components/TreeNodeComponent.tsx
// NEW FILE - Extracted component
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    FiFolder, FiFileText, FiMinusSquare, FiPlusSquare, FiExternalLink,
    FiCode, FiDatabase, FiLayout, FiBarChart, FiSettings, FiZap, FiCpu,
    // Add more specific icons as needed
    FiBookOpen, // for notebooks
    FiBox, // for src/data
    FiSliders, // for src/models
    FiGitBranch, // for root
    FiFile, // default file
    FiTerminal, // for pipelines
    FiCheckSquare // for tests
} from 'react-icons/fi';
import { VscJson, VscMarkdown, VscNotebook, VscSymbolFile, VscSymbolNamespace } from "react-icons/vsc"; // More specific icons
import { DiPython, DiReact, DiHtml5, DiCss3, DiJavascript1 } from "react-icons/di"; // Tech icons


interface TreeNode {
    name: string;
    path: string; // Full path from root
    type: 'folder' | 'file';
    comment?: string;
    children?: TreeNode[];
}

// Helper to get appropriate icon
const getNodeIcon = (node: TreeNode, isOpen: boolean): React.ReactNode => {
    const nameLower = node.name.toLowerCase();
    const isFolder = node.type === 'folder';

    if (isFolder) {
        // Specific folder icons
        if (nameLower === 'src') return <FiBox size={16} />;
        if (nameLower === 'api') return <FiZap size={16} />;
        if (nameLower === 'frontend') return <FiLayout size={16} />;
        if (nameLower === 'notebooks') return <FiBookOpen size={16} />;
        if (nameLower === 'data') return <FiDatabase size={16} />;
        if (nameLower === 'models') return <FiCpu size={16} />;
        if (nameLower === 'pipelines') return <FiTerminal size={16} />;
        if (nameLower === 'tests') return <FiCheckSquare size={16} />;
        if (nameLower === 'evaluation') return <FiBarChart size={16} />;
        if (nameLower === 'database') return <FiDatabase size={16} />;
        if (nameLower === 'components') return <DiReact size={18} />; // React icon for components
        if (nameLower === 'pages') return <FiLayout size={16} />;
        // Default folder icon
        return <FiFolder size={16} className={isOpen ? 'text-primary' : 'text-text-muted'} />;
    } else {
        // Specific file icons by extension
        if (nameLower.endsWith('.py')) return <DiPython size={18} />;
        if (nameLower.endsWith('.ipynb')) return <VscNotebook size={16} />;
        if (nameLower.endsWith('.tsx')) return <DiReact size={18} />;
        if (nameLower.endsWith('.ts')) return <DiJavascript1 size={18} />; // Close enough for TS
        if (nameLower.endsWith('.md')) return <VscMarkdown size={16} />;
        if (nameLower.endsWith('.json')) return <VscJson size={16} />;
        if (nameLower.endsWith('.html')) return <DiHtml5 size={18} />;
        if (nameLower.endsWith('.css') || nameLower.endsWith('.cjs')) return <DiCss3 size={18} />;
        if (nameLower.endsWith('.txt') || nameLower.endsWith('.example') || nameLower.endsWith('.ini') || nameLower.endsWith('.gitignore')) return <VscSymbolFile size={16} />;
        // Default file icon
        return <FiFileText size={16} className="text-text-muted" />;
    }
};

interface TreeNodeComponentProps {
    node: TreeNode;
    level: number;
    githubBaseUrl: string;
    searchTerm: string;
    allExpanded: boolean; // New prop
    filterVisible: boolean; // Whether this node should be visible based on parent filter
}

const TreeNodeComponent: React.FC<TreeNodeComponentProps> = ({ node, level, githubBaseUrl, searchTerm, allExpanded, filterVisible }) => {
    const [isOpen, setIsOpen] = useState(level < 1); // Default open level 0
    const isFolder = node.type === 'folder';
    const toggleOpen = () => { if (isFolder) setIsOpen(!isOpen); };

    // Update open state based on prop
    useEffect(() => {
        if (isFolder) {
             setIsOpen(allExpanded);
        }
    }, [allExpanded, isFolder]);

    // Determine visibility based on search term
    const nameLower = node.name.toLowerCase();
    const searchTermLower = searchTerm.toLowerCase();
    const matchesSearch = searchTermLower === '' || nameLower.includes(searchTermLower);

    // If the node itself matches, or if any child matches, it should be visible (unless filtered by parent)
    const hasVisibleChild = isFolder && node.children?.some(child => child.name.toLowerCase().includes(searchTermLower)) || false;
    const shouldBeVisible = filterVisible && (matchesSearch || hasVisibleChild);

    const linkUrl = `${githubBaseUrl}/${node.path}`;

    const icon = getNodeIcon(node, isOpen);
    const textColor = isFolder ? 'text-text-primary font-medium' : 'text-text-secondary';

    const nodeContent = (
        <div
            className={`flex items-center py-1 group hover:bg-border-color/20 rounded ${isFolder ? 'cursor-pointer' : ''}`}
            onClick={toggleOpen} role={isFolder ? 'button' : undefined} tabIndex={isFolder ? 0 : -1}
            onKeyDown={(e) => { if (isFolder && (e.key === 'Enter' || e.key === ' ')) toggleOpen(); }}
            style={{ paddingLeft: `${level * 20}px` }}
            title={node.path} // Show full path on hover
        >
            <span className="w-[18px] text-center mr-1 flex-shrink-0 text-text-muted">
                {isFolder && (isOpen ? <FiMinusSquare size={14} /> : <FiPlusSquare size={14} />)}
            </span>
            <span className={`mr-2 flex-shrink-0 ${isFolder && isOpen ? 'text-primary' : 'text-text-muted'}`}>{icon}</span>

            <span className={`${textColor} ${!matchesSearch && searchTerm ? 'opacity-50' : ''}`}>
                {node.name}
            </span>

            {node.comment && (
                <span className="ml-3 text-xs text-text-muted italic opacity-0 group-hover:opacity-100 transition-opacity hidden md:inline">
                    // {node.comment}
                </span>
            )}
            {/* External link icon */}
            <a
                href={linkUrl} target="_blank" rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()} // Prevent toggleOpen on icon click
                className="ml-auto mr-2 text-text-muted hover:text-primary opacity-0 group-hover:opacity-100 transition-opacity"
                title={`View ${node.name} on GitHub`}
            >
                <FiExternalLink size={14} />
            </a>
        </div>
    );

    // Don't render if filtered out by parent
    if (!filterVisible) return null;

    // Render node, but hide if search doesn't match and no children match
    // Render children recursively, passing down visibility state
    return (
        <>
            {/* Apply opacity if this node doesn't match but has visible children */}
            <div className={`${!matchesSearch && hasVisibleChild ? 'opacity-60' : ''}`}>
                {nodeContent}
            </div>

            {isFolder && node.children && (
                 <AnimatePresence initial={false}>
                     {isOpen && (
                         <motion.div
                             initial={{ height: 0, opacity: 0 }}
                             animate={{ height: 'auto', opacity: 1 }}
                             exit={{ height: 0, opacity: 0 }}
                             transition={{ duration: 0.2 }}
                             className="border-l border-dashed border-border-color ml-[9px] pl-2 overflow-hidden" // Adjusted margin/padding
                         >
                             {node.children.map((child, index) => (
                                 <TreeNodeComponent
                                     key={`${child.path}-${index}`}
                                     node={child}
                                     level={level + 1}
                                     githubBaseUrl={githubBaseUrl}
                                     searchTerm={searchTerm}
                                     allExpanded={allExpanded}
                                     // Child is visible if its own name matches search OR if the current folder matches (so all children show)
                                     filterVisible={shouldBeVisible} // Pass down visibility based on current node's status
                                 />
                             ))}
                         </motion.div>
                     )}
                 </AnimatePresence>
             )}
        </>
    );
};


export default TreeNodeComponent;