// frontend/src/components/HighlightItem.tsx
// NEW FILE - Extracted component
import React from 'react';
import { motion } from 'framer-motion';
import { FiGithub } from 'react-icons/fi';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { atomOneDark } from 'react-syntax-highlighter/dist/esm/styles/hljs'; // Or choose another theme

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
                {/* Use dangerouslySetInnerHTML if description contains simple <code> tags */}
                <p className="text-sm text-text-muted" dangerouslySetInnerHTML={{ __html: description }}></p>
            </div>

            {/* Code Snippet */}
            {codeSnippet && (
                <div className="mb-5 max-h-48 overflow-y-auto rounded-md bg-gray-900 text-xs border border-border-color">
                    <SyntaxHighlighter
                        language={language}
                        style={atomOneDark} // Apply the theme
                        customStyle={{ margin: 0, padding: '0.75rem 1rem', background: 'transparent' }} // Override defaults
                        showLineNumbers={false} // Optional: show line numbers
                        wrapLines={true}
                        wrapLongLines={true} // Helps prevent horizontal scroll
                    >
                        {codeSnippet.trim()}
                    </SyntaxHighlighter>
                </div>
            )}

            {/* Footer Action */}
            {githubUrl && (
                <div className="mt-auto pt-3 border-t border-border-color/50 text-right">
                    <a
                        href={githubUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs inline-flex items-center gap-1.5 text-text-muted hover:text-primary transition-colors font-medium"
                    >
                        View on GitHub <FiGithub size={14} />
                    </a>
                </div>
            )}
        </motion.div>
    );
};

export default HighlightItem;