// frontend/src/components/HighlightItem.tsx
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { FiGithub, FiCopy, FiCheck, FiMaximize2, FiExternalLink } from 'react-icons/fi';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark as SyntaxHighlightTheme } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface HighlightItemProps {
    icon: React.ReactNode;
    title: string;
    description: string;
    path?: string;
    githubUrl?: string;
    codeSnippet?: string;
    language?: string;
    listView?: boolean;
    expandedView?: boolean;
}

const HighlightItem: React.FC<HighlightItemProps> = ({
    icon, 
    title, 
    description, 
    path, 
    githubUrl, 
    codeSnippet, 
    language = 'python',
    listView = false,
    expandedView = false
}) => {
    const [copied, setCopied] = useState(false);
    const [isHovered, setIsHovered] = useState(false);

    const handleCopy = (e: React.MouseEvent) => {
        e.stopPropagation(); // Prevent triggering parent click handlers
        if (codeSnippet) {
            navigator.clipboard.writeText(codeSnippet.trim())
                .then(() => {
                    setCopied(true);
                    setTimeout(() => setCopied(false), 2000); // Reset after 2 seconds
                })
                .catch(err => console.error('Failed to copy text: ', err));
        }
    };

    // If in list view mode, render a more compact horizontal layout
    if (listView) {
        return (
            <motion.div
                className="bg-surface/80 p-4 rounded-xl border border-border-color/70 shadow-lg transform transition duration-300 ease-out hover:border-primary/50 hover:shadow-primary/10 focus-within:ring-2 focus-within:ring-primary focus-within:border-primary/50 cursor-pointer"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                whileHover={{ scale: 1.01 }}
                onMouseEnter={() => setIsHovered(true)}
                onMouseLeave={() => setIsHovered(false)}
            >
                <div className="flex items-center gap-4">
                    <span className="text-primary text-2xl flex-shrink-0">{icon}</span>
                    <div className="flex-grow min-w-0">
                        <h4 className="text-lg font-semibold text-text-primary truncate">{title}</h4>
                        {path && (
                            <p className="text-xs font-mono text-primary/70 truncate">
                                {path}
                            </p>
                        )}
                    </div>
                    <div className="flex-shrink-0 flex items-center gap-2">
                        {githubUrl && (
                            <a 
                                href={githubUrl} 
                                target="_blank" 
                                rel="noopener noreferrer" 
                                onClick={(e) => e.stopPropagation()}
                                className="p-2 text-text-muted hover:text-primary transition-colors rounded-full hover:bg-border-color/30"
                            >
                                <FiExternalLink size={18} />
                            </a>
                        )}
                        <div className="p-2 text-text-muted/70 rounded-full">
                            <FiMaximize2 size={18} />
                        </div>
                    </div>
                </div>
            </motion.div>
        );
    }

    // Standard card view (default)
    return (
        <motion.div
            // Card styling
            className={`bg-surface/80 p-6 rounded-xl border border-border-color/70 shadow-lg h-full flex flex-col transform transition duration-300 ease-out hover:border-primary/50 hover:shadow-primary/10 focus-within:ring-2 focus-within:ring-primary focus-within:border-primary/50 ${!expandedView ? 'cursor-pointer' : ''}`}
            // Animation props
            initial={{ opacity: 0, y: 25 }}
            animate={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.2 }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
        >
            {/* Header */}
            <div className="flex items-center mb-4">
                <span className="text-primary mr-4 text-2xl flex-shrink-0">{icon}</span>
                <h4 className="text-lg font-semibold text-text-primary flex-1 leading-snug">{title}</h4>
                
                {!expandedView && (
                    <motion.div 
                        className="text-text-muted/60 hover:text-primary p-1 rounded-full hover:bg-border-color/30"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: isHovered ? 1 : 0 }}
                        transition={{ duration: 0.2 }}
                    >
                        <FiMaximize2 size={18} />
                    </motion.div>
                )}
            </div>

            {/* Content */}
            <div className="flex-grow mb-5">
                {path && (
                    <p className="text-xs font-mono bg-background inline-block px-2 py-1 rounded mb-3 text-primary/80 border border-border-color break-all">
                        {path}
                    </p>
                )}
                <div className={`text-sm text-text-muted prose prose-sm prose-invert max-w-none prose-p:text-text-muted prose-code:text-primary/80 prose-code:bg-background/50 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-xs prose-code:font-mono ${!expandedView && 'line-clamp-3'}`}>
                     <p dangerouslySetInnerHTML={{ __html: description }}></p>
                </div>
            </div>

            {/* Code Snippet */}
            {codeSnippet && (
                <div className={`mb-5 ${expandedView ? '' : 'max-h-60'} overflow-y-auto rounded-lg bg-black/60 text-xs border border-border-color/60 relative group/snippet p-4 shadow-inner`}>
                     {/* Copy Button */}
                     <button
                         onClick={handleCopy}
                         className="absolute top-2.5 right-2.5 p-1.5 bg-gray-800/70 hover:bg-gray-700/90 rounded text-gray-300 hover:text-white opacity-0 group-hover/snippet:opacity-100 transition-all focus:opacity-100 outline-none focus-visible:ring-1 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-black/60 z-10"
                         aria-label={copied ? "Copied!" : "Copy code snippet"}
                     >
                         {copied ? <FiCheck size={14} className="text-green-400"/> : <FiCopy size={14} />}
                     </button>
                    <SyntaxHighlighter
                        language={language}
                        style={SyntaxHighlightTheme}
                        customStyle={{
                            margin: 0,
                            padding: '0.1rem 0.5rem 0.1rem 0.5rem',
                            paddingRight: '3rem',
                            background: 'transparent',
                            fontSize: '0.8rem',
                            lineHeight: '1.4',
                            whiteSpace: 'pre-wrap',
                            wordBreak: 'break-all',
                            maxHeight: expandedView ? 'none' : '100%'
                        }}
                        showLineNumbers={expandedView}
                        wrapLines={true}
                        wrapLongLines={true}
                    >
                        {codeSnippet.trim()}
                    </SyntaxHighlighter>
                </div>
            )}

            {/* Footer Action */}
            {githubUrl && !expandedView && (
                <div className="mt-auto pt-3 border-t border-border-color/50 text-right">
                    <a 
                        href={githubUrl} 
                        target="_blank" 
                        rel="noopener noreferrer" 
                        onClick={(e) => e.stopPropagation()}
                        className="text-xs inline-flex items-center gap-1.5 text-text-muted hover:text-primary transition-colors font-medium focus:outline-none focus-visible:ring-1 focus-visible:ring-primary rounded-sm"
                    >
                        View on GitHub <FiGithub size={14} />
                    </a>
                </div>
            )}
        </motion.div>
    );
};

export default HighlightItem;