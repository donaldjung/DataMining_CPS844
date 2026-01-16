interface OutputDisplayProps {
  output: string;
  title?: string;
}

export default function OutputDisplay({ output, title = 'Output' }: OutputDisplayProps) {
  return (
    <div className="terminal my-4 overflow-hidden">
      <div className="terminal-header">
        <div className="terminal-dot red"></div>
        <div className="terminal-dot yellow"></div>
        <div className="terminal-dot green"></div>
        <span className="ml-4 text-sm text-gray-400">{title}</span>
      </div>
      <div className="p-4 overflow-x-auto">
        <pre 
          className="text-sm leading-relaxed whitespace-pre-wrap"
          style={{ color: '#4ade80' }}
        >
          {output}
        </pre>
      </div>
    </div>
  );
}
