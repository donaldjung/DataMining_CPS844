import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import mermaid from 'mermaid';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Initialize mermaid with dark theme
mermaid.initialize({
  startOnLoad: true,
  theme: 'dark',
  themeVariables: {
    primaryColor: '#3b82f6',
    primaryTextColor: '#e2e8f0',
    primaryBorderColor: '#3b82f6',
    lineColor: '#64748b',
    secondaryColor: '#8b5cf6',
    tertiaryColor: '#0f172a',
    background: '#020617',
    mainBkg: '#0f172a',
    nodeBorder: '#3b82f6',
    clusterBkg: '#1e293b',
    titleColor: '#f97316',
    edgeLabelBackground: '#0f172a',
  },
  flowchart: {
    htmlLabels: true,
    curve: 'basis',
  },
});

export { mermaid };
