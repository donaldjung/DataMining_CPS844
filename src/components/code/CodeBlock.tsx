import { useState } from 'react';
import { Copy, Check } from 'lucide-react';

interface CodeBlockProps {
  code: string;
  language?: string;
  title?: string;
  showLineNumbers?: boolean;
}

const COLORS = {
  keyword: '#c084fc',
  function: '#60a5fa',
  string: '#4ade80',
  number: '#f97316',
  comment: '#64748b',
  className: '#22d3ee',
  operator: '#f472b6',
  decorator: '#fbbf24',
  builtin: '#a78bfa',
  default: '#e2e8f0',
};

export default function CodeBlock({ code, language = 'python', title, showLineNumbers = true }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const copyToClipboard = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const highlightPython = (line: string): (JSX.Element | string)[] => {
    const tokens: (JSX.Element | string)[] = [];
    let remaining = line;
    let keyIndex = 0;

    // Python keywords
    const keywords = ['import', 'from', 'def', 'class', 'return', 'if', 'elif', 'else', 'for', 'while', 'in', 'not', 'and', 'or', 'is', 'None', 'True', 'False', 'try', 'except', 'finally', 'with', 'as', 'lambda', 'yield', 'raise', 'pass', 'break', 'continue', 'global', 'nonlocal', 'assert', 'async', 'await'];
    const builtins = ['print', 'len', 'range', 'list', 'dict', 'set', 'tuple', 'str', 'int', 'float', 'bool', 'type', 'isinstance', 'enumerate', 'zip', 'map', 'filter', 'sorted', 'sum', 'min', 'max', 'abs', 'round', 'open', 'input', 'super', 'self'];

    const patterns: [RegExp, string][] = [
      [/^(#.*)/, 'comment'],
      [/^(@\w+)/, 'decorator'],
      [/^("""[\s\S]*?"""|'''[\s\S]*?'''|"[^"]*"|'[^']*')/, 'string'],
      [/^(\d+\.?\d*|\.\d+)/, 'number'],
      [/^(def|class)\s+(\w+)/, 'defclass'],
      [/^(\w+)(\s*\()/, 'funcall'],
      [/^([+\-*/%=<>!&|^~]+|:)/, 'operator'],
      [/^(\w+)/, 'identifier'],
      [/^(\s+)/, 'whitespace'],
      [/^(.)/, 'other'],
    ];

    while (remaining.length > 0) {
      let matched = false;

      for (const [pattern, type] of patterns) {
        const match = remaining.match(pattern);
        if (match) {
          matched = true;
          
          if (type === 'comment') {
            tokens.push(<span key={keyIndex++} style={{ color: COLORS.comment, fontStyle: 'italic' }}>{match[1]}</span>);
            remaining = remaining.slice(match[1].length);
          } else if (type === 'decorator') {
            tokens.push(<span key={keyIndex++} style={{ color: COLORS.decorator }}>{match[1]}</span>);
            remaining = remaining.slice(match[1].length);
          } else if (type === 'string') {
            tokens.push(<span key={keyIndex++} style={{ color: COLORS.string }}>{match[1]}</span>);
            remaining = remaining.slice(match[1].length);
          } else if (type === 'number') {
            tokens.push(<span key={keyIndex++} style={{ color: COLORS.number }}>{match[1]}</span>);
            remaining = remaining.slice(match[1].length);
          } else if (type === 'defclass') {
            tokens.push(<span key={keyIndex++} style={{ color: COLORS.keyword }}>{match[1]}</span>);
            tokens.push(' ');
            tokens.push(<span key={keyIndex++} style={{ color: COLORS.className }}>{match[2]}</span>);
            remaining = remaining.slice(match[0].length);
          } else if (type === 'funcall') {
            const funcName = match[1];
            if (builtins.includes(funcName)) {
              tokens.push(<span key={keyIndex++} style={{ color: COLORS.builtin }}>{funcName}</span>);
            } else {
              tokens.push(<span key={keyIndex++} style={{ color: COLORS.function }}>{funcName}</span>);
            }
            tokens.push(<span key={keyIndex++} style={{ color: COLORS.operator }}>{match[2]}</span>);
            remaining = remaining.slice(match[0].length);
          } else if (type === 'operator') {
            tokens.push(<span key={keyIndex++} style={{ color: COLORS.operator }}>{match[1]}</span>);
            remaining = remaining.slice(match[1].length);
          } else if (type === 'identifier') {
            const word = match[1];
            if (keywords.includes(word)) {
              tokens.push(<span key={keyIndex++} style={{ color: COLORS.keyword }}>{word}</span>);
            } else if (builtins.includes(word)) {
              tokens.push(<span key={keyIndex++} style={{ color: COLORS.builtin }}>{word}</span>);
            } else {
              tokens.push(<span key={keyIndex++} style={{ color: COLORS.default }}>{word}</span>);
            }
            remaining = remaining.slice(word.length);
          } else if (type === 'whitespace') {
            tokens.push(match[1]);
            remaining = remaining.slice(match[1].length);
          } else {
            tokens.push(match[1]);
            remaining = remaining.slice(match[1].length);
          }
          break;
        }
      }

      if (!matched) {
        tokens.push(remaining[0]);
        remaining = remaining.slice(1);
      }
    }

    return tokens;
  };

  const lines = code.split('\n');

  return (
    <div className="terminal my-4 overflow-hidden">
      <div className="terminal-header">
        <div className="terminal-dot red"></div>
        <div className="terminal-dot yellow"></div>
        <div className="terminal-dot green"></div>
        <span className="ml-4 text-sm text-gray-400">{title || `${language}.py`}</span>
        <button
          onClick={copyToClipboard}
          className="ml-auto p-1.5 rounded hover:bg-white/10 transition-colors"
          title="Copy code"
        >
          {copied ? (
            <Check size={14} style={{ color: '#4ade80' }} />
          ) : (
            <Copy size={14} className="text-gray-400" />
          )}
        </button>
      </div>
      <div className="p-4 overflow-x-auto">
        <pre className="text-sm leading-relaxed">
          {lines.map((line, index) => (
            <div key={index} className="flex">
              {showLineNumbers && (
                <span 
                  className="select-none mr-4 text-right w-8"
                  style={{ color: '#475569' }}
                >
                  {index + 1}
                </span>
              )}
              <code>{highlightPython(line)}</code>
            </div>
          ))}
        </pre>
      </div>
    </div>
  );
}
