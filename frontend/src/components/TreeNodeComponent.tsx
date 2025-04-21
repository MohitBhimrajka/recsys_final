// frontend/src/components/TreeNodeComponent.tsx
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    FiFolder, FiFileText, FiMinusSquare, FiPlusSquare, FiExternalLink,
    FiCode, FiDatabase, FiLayout, FiBarChart, FiSettings, FiZap, FiCpu,
    FiBookOpen, FiBox, FiSliders, FiGitBranch, FiFile, FiTerminal, FiCheckSquare,
    FiHardDrive
} from 'react-icons/fi';
import { VscJson, VscMarkdown, VscNotebook, VscSymbolFile, VscVmActive } from "react-icons/vsc";
import { DiPython, DiReact, DiHtml5, DiCss3, DiJavascript1, DiTerminal } from "react-icons/di";
import { FaGitAlt, FaDocker, FaDatabase, FaNodeJs } from "react-icons/fa";
import { SiJson, SiMarkdown, SiJupyter, SiTypescript, SiFastapi, SiVite, SiTailwindcss, SiEslint, SiPostcss, SiPnpm, SiGithubactions } from "react-icons/si";

interface TreeNode {
    name: string;
    path: string;
    type: 'folder' | 'file';
    comment?: string;
    children?: TreeNode[];
}

// Icon Mapping Function (Unchanged from Phase 4)
const getNodeIcon = (node: TreeNode, isOpen: boolean): React.ReactNode => {
    const nameLower = node.name.toLowerCase();
    const isFolder = node.type === 'folder';
    const iconSize = 16;
    const iconColor = isOpen ? 'text-primary' : 'text-text-muted';

    if (isFolder) {
        if (nameLower === 'src') return <FiBox size={iconSize} className={iconColor} title="Source Code Root"/>;
        if (nameLower === 'api') return <SiFastapi size={iconSize} className={iconColor} title="FastAPI Backend"/>;
        if (nameLower === 'frontend') return <SiVite size={iconSize} className={iconColor} title="Vite Frontend"/>;
        if (nameLower === 'components') return <DiReact size={iconSize + 2} className={iconColor} title="React Components"/>;
        if (nameLower === 'pages') return <FiLayout size={iconSize} className={iconColor} title="Frontend Pages"/>;
        if (nameLower === 'services') return <FiZap size={iconSize} className={iconColor} title="API Services / Utilities"/>;
        if (nameLower === 'models') return <FiCpu size={iconSize} className={iconColor} title="Recommendation Models"/>;
        if (nameLower === 'data') return <FaDatabase size={iconSize} className={iconColor} title="Data Processing/Loading"/>;
        if (nameLower === 'database') return <FiDatabase size={iconSize} className={iconColor} title="Database Interaction"/>;
        if (nameLower === 'pipelines') return <DiTerminal size={iconSize + 2} className={iconColor} title="Execution Pipelines"/>;
        if (nameLower === 'notebooks') return <SiJupyter size={iconSize} className={iconColor} title="Jupyter Notebooks"/>;
        if (nameLower === 'tests' || nameLower === '__tests__') return <FiCheckSquare size={iconSize} className={iconColor} title="Tests"/>;
        if (nameLower === 'reports') return <FiBarChart size={iconSize} className={iconColor} title="Reports"/>;
        if (nameLower === 'public') return <FiHardDrive size={iconSize} className={iconColor} title="Public Assets"/>;
        if (nameLower === 'node_modules') return <FaNodeJs size={iconSize} className={iconColor} title="Node Modules"/>;
        if (nameLower === '.vscode') return <FiSettings size={iconSize} className={iconColor} title="VS Code Settings"/>;
        if (nameLower === '.github') return <SiGithubactions size={iconSize} className={iconColor} title="GitHub Actions"/>;
        return <FiFolder size={iconSize} className={iconColor} title="Folder"/>;
    } else {
        if (nameLower.endsWith('.py')) return <DiPython size={iconSize + 2} className="text-blue-400" title="Python File"/>;
        if (nameLower.endsWith('.ipynb')) return <SiJupyter size={iconSize} className="text-orange-500" title="Jupyter Notebook"/>;
        if (nameLower.endsWith('.tsx')) return <DiReact size={iconSize + 2} className="text-cyan-400" title="React TSX File"/>;
        if (nameLower.endsWith('.ts')) return <SiTypescript size={iconSize} className="text-blue-500" title="TypeScript File"/>;
        if (nameLower.endsWith('.js')) return <DiJavascript1 size={iconSize + 2} className="text-yellow-400" title="JavaScript File"/>;
        if (nameLower.endsWith('.md')) return <SiMarkdown size={iconSize} className="text-gray-400" title="Markdown File"/>;
        if (nameLower.endsWith('.json')) return <SiJson size={iconSize} className="text-yellow-600" title="JSON File"/>;
        if (nameLower.endsWith('.html')) return <DiHtml5 size={iconSize + 2} className="text-orange-600" title="HTML File"/>;
        if (nameLower.endsWith('.css')) return <DiCss3 size={iconSize + 2} className="text-blue-600" title="CSS File"/>;
        if (nameLower.endsWith('.env') || nameLower.endsWith('.env.example')) return <VscSymbolFile size={iconSize} className="text-green-500" title="Environment Variables"/>;
        if (nameLower === 'dockerfile') return <FaDocker size={iconSize} className="text-blue-500" title="Dockerfile"/>;
        if (nameLower.includes('tailwind.config')) return <SiTailwindcss size={iconSize} className="text-teal-400" title="Tailwind Config"/>;
        if (nameLower.includes('vite.config')) return <SiVite size={iconSize} className="text-purple-500" title="Vite Config"/>;
        if (nameLower.includes('postcss.config')) return <SiPostcss size={iconSize} className="text-orange-500" title="PostCSS Config"/>;
        if (nameLower.includes('eslint')) return <SiEslint size={iconSize} className="text-purple-600" title="ESLint Config"/>;
        if (nameLower === 'readme.md') return <FiBookOpen size={iconSize} className="text-blue-400" title="README"/>;
        if (nameLower.endsWith('.lock') || nameLower === 'pnpm-workspace.yaml') return <SiPnpm size={iconSize} className="text-orange-500" title="Package Lock/Workspace"/>;
        if (nameLower === '.gitignore' || nameLower === '.gitattributes') return <FaGitAlt size={iconSize} className="text-orange-600" title="Git File"/>;
        if (nameLower === 'requirements.txt') return <VscSymbolFile size={iconSize} className="text-gray-400" title="Python Requirements"/>;
        if (nameLower === 'pytest.ini') return <VscSymbolFile size={iconSize} className="text-gray-400" title="Pytest Config"/>;
        return <FiFileText size={iconSize} className="text-text-muted" title="File"/>;
    }
};

interface TreeNodeComponentProps {
    node: TreeNode;
    level: number;
    githubBaseUrl: string;
    searchTerm: string;
    allExpanded: boolean; // Controlled from parent
    filterVisible: boolean; // Whether parent allows visibility
}

const TreeNodeComponent: React.FC<TreeNodeComponentProps> = ({ node, level, githubBaseUrl, searchTerm, allExpanded, filterVisible }) => {
    // *** MODIFICATION: Default open state now relies on allExpanded prop initially ***
    const [isOpen, setIsOpen] = useState(allExpanded);
    const isFolder = node.type === 'folder';
    const toggleOpen = () => { if (isFolder) setIsOpen(!isOpen); };

    // Sync open state with the external `allExpanded` prop
    useEffect(() => {
        if (isFolder) {
             setIsOpen(allExpanded);
        }
        // Do not automatically collapse if already open when allExpanded becomes false
        // Let user manually collapse or use the Collapse All button
    }, [allExpanded, isFolder]);

    // Search logic (Unchanged)
    const nameLower = node.name.toLowerCase();
    const searchTermLower = searchTerm.toLowerCase();
    const matchesSearch = searchTermLower === '' || nameLower.includes(searchTermLower) || (node.comment?.toLowerCase() || '').includes(searchTermLower);
    const hasMatchingChild = (currentNode: TreeNode): boolean => { /* ... implementation unchanged ... */
        if (!currentNode.children) return false;
        return currentNode.children.some(child =>
            child.name.toLowerCase().includes(searchTermLower) ||
            (child.comment?.toLowerCase() || '').includes(searchTermLower) ||
            hasMatchingChild(child)
        );
     };
    const hasVisibleChild = isFolder && node.children ? hasMatchingChild(node) : false;
    const shouldBeVisible = filterVisible && (matchesSearch || hasVisibleChild);

    if (!shouldBeVisible && searchTerm !== '') return null;

    const linkUrl = `${githubBaseUrl}/${node.path}`;
    const icon = getNodeIcon(node, isOpen);
    const textColor = isFolder ? 'text-text-primary font-medium' : 'text-text-secondary';

    // Node Content JSX (Unchanged)
    const nodeContent = (
        <div className={`flex items-center py-1 group hover:bg-border-color/20 rounded relative outline-none ${isFolder ? 'cursor-pointer' : ''} focus-visible:ring-1 focus-visible:ring-primary`} onClick={toggleOpen} role={isFolder ? 'button' : undefined} tabIndex={isFolder ? 0 : -1} onKeyDown={(e) => { if (isFolder && (e.key === 'Enter' || e.key === ' ')) toggleOpen(); }} style={{ paddingLeft: `${level * 24}px` }} title={node.path} >
             {level > 0 && Array.from({ length: level }).map((_, i) => ( <span key={i} className="absolute top-0 bottom-0 left-0 w-px bg-border-color/30" style={{ transform: `translateX(${(i * 24) + 10}px)` }}></span> ))}
             <span className="w-[18px] text-center mr-1 flex-shrink-0 text-text-muted/80 z-10"> {isFolder && (isOpen ? <FiMinusSquare size={14} /> : <FiPlusSquare size={14} />)} </span>
             <span className={`mr-2 flex-shrink-0 flex items-center z-10`}>{icon}</span>
             <span className={`${textColor} z-10 ${!matchesSearch && searchTerm ? 'opacity-60' : ''}`}> {node.name} </span>
             {node.comment && ( <span className="ml-3 text-xs text-text-muted/80 italic opacity-0 group-hover:opacity-100 transition-opacity hidden md:inline z-10"> {node.comment} </span> )}
             <a href={linkUrl} target="_blank" rel="noopener noreferrer" onClick={(e) => e.stopPropagation()} className="ml-auto mr-2 text-text-muted hover:text-primary opacity-0 group-hover:opacity-100 transition-opacity z-10 focus:outline-none focus-visible:ring-1 focus-visible:ring-primary rounded-sm" title={`View ${node.name} on GitHub`} > <FiExternalLink size={14} /> </a>
        </div>
    );

    // Render Logic (Unchanged)
    const selfRender = ( <div className={`${!matchesSearch && hasVisibleChild ? 'opacity-70' : ''}`}> {nodeContent} </div> )

    return (
        <>
            {(matchesSearch || hasVisibleChild) && selfRender}
            {isFolder && node.children && (
                 <AnimatePresence initial={false}>
                     {isOpen && (
                         <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.2, ease: 'easeInOut' }} className="overflow-hidden" >
                             {node.children.map((child, index) => ( <TreeNodeComponent key={`${child.path}-${index}`} node={child} level={level + 1} githubBaseUrl={githubBaseUrl} searchTerm={searchTerm} allExpanded={allExpanded} filterVisible={true} /> ))}
                         </motion.div>
                     )}
                 </AnimatePresence>
             )}
        </>
    );
};

export default TreeNodeComponent;