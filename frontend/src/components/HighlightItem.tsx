// frontend/src/components/HighlightItem.tsx
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { FiGithub, FiCopy, FiCheck } from 'react-icons/fi'; // Added Copy/Check icons
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
}

const HighlightItem: React.FC<HighlightItemProps> = ({
    icon, title, description, path, githubUrl, codeSnippet, language = 'python'
}) => {
    const [copied, setCopied] = useState(false);

    const handleCopy = () => {
        if (codeSnippet) {
            navigator.clipboard.writeText(codeSnippet.trim())
                .then(() => {
                    setCopied(true);
                    setTimeout(() => setCopied(false), 2000); // Reset after 2 seconds
                })
                .catch(err => console.error('Failed to copy text: ', err));
        }
    };

    return (
        <motion.div
            className="bg-surface p-6 rounded-xl border border-border-color shadow-lg h-full flex flex-col transform transition duration-300 hover:border-primary/50 hover:-translate-y-1 hover:shadow-primary/10"
            initial={{ opacity: 0, y: 15 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.2 }}
            transition={{ duration: 0.4, ease: 'easeOut' }}
        >
            {/* Header */}
            <div className="flex items-center mb-4">
                <span className="text-primary mr-4 text-2xl flex-shrink-0">{icon}</span>
                <h4 className="text-lg font-semibold text-text-primary flex-1 leading-snug">{title}</h4>
            </div>

            {/* Content */}
            <div className="flex-grow mb-5">
                {path && (
                    <p className="text-xs font-mono bg-background inline-block px-2 py-1 rounded mb-3 text-primary/80 border border-border-color break-all">
                        {path}
                    </p>
                )}
                <div className="text-sm text-text-muted prose prose-sm prose-invert max-w-none prose-p:text-text-muted prose-code:text-primary/80 prose-code:bg-background/50 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-xs prose-code:font-mono">
                     <p dangerouslySetInnerHTML={{ __html: description }}></p>
                </div>
            </div>

            {/* Code Snippet */}
            {codeSnippet && (
                <div className="mb-5 max-h-56 overflow-y-auto rounded-md bg-black/50 text-xs border border-border-color/70 relative group/snippet">
                     {/* Copy Button */}
                     <button
                         onClick={handleCopy}
                         className="absolute top-2 right-2 p-1.5 bg-gray-700/60 hover:bg-gray-600/80 rounded text-gray-300 hover:text-white opacity-0 group-hover/snippet:opacity-100 transition-all focus:opacity-100 outline-none focus-visible:ring-1 focus-visible:ring-primary z-10" // Ensure button is interactive
                         aria-label={copied ? "Copied!" : "Copy code snippet"}
                     >
                         {copied ? <FiCheck size={14} className="text-green-400"/> : <FiCopy size={14} />}
                     </button>
                    <SyntaxHighlighter
                        language={language}
                        style={SyntaxHighlightTheme}
                        customStyle={{
                            margin: 0, padding: '0.8rem 1rem', paddingRight: '2.5rem', // Make space for button
                            background: 'transparent', fontSize: '0.78rem', whiteSpace: 'pre-wrap', wordBreak: 'break-all'
                        }}
                        showLineNumbers={false} wrapLines={true} wrapLongLines={true}
                    >
                        {codeSnippet.trim()}
                    </SyntaxHighlighter>
                </div>
            )}

            {/* Footer Action */}
            {githubUrl && (
                <div className="mt-auto pt-3 border-t border-border-color/50 text-right">
                    <a href={githubUrl} target="_blank" rel="noopener noreferrer" className="text-xs inline-flex items-center gap-1.5 text-text-muted hover:text-primary transition-colors font-medium focus:outline-none focus-visible:ring-1 focus-visible:ring-primary rounded-sm">
                        View on GitHub <FiGithub size={14} />
                    </a>
                </div>
            )}
        </motion.div>
    );
};

export default HighlightItem;