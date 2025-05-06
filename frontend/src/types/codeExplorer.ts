import { ReactNode } from 'react';

// Define the categories of code available for filtering
export type CodeCategory = 'all' | 'python' | 'typescript' | 'frontend' | 'backend' | 'data' | 'models' | 'evaluation';

// Display information for categories
export interface CategoryInfo {
  name: string;
  icon: ReactNode;
}

// Category display info type
export type CategoryDisplayInfo = Record<CodeCategory, CategoryInfo>;

// Code item structure
export interface CodeItem {
  id: string;
  icon: ReactNode;
  title: string;
  path: string;
  githubUrl: string;
  description: string;
  codeSnippet?: string;
  language: string;
  category: CodeCategory[];
  isFeatured?: boolean;
}

// Tree node structure for file explorer
export interface TreeNode {
  name: string;
  path: string;
  type: 'folder' | 'file';
  comment?: string;
  children?: TreeNode[];
} 