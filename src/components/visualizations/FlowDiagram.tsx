import { useEffect, useRef } from 'react';
import { mermaid } from '../../lib/utils';

interface FlowDiagramProps {
  chart: string;
  title?: string;
}

export default function FlowDiagram({ chart, title }: FlowDiagramProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const renderDiagram = async () => {
      if (containerRef.current) {
        containerRef.current.innerHTML = '';
        const id = `mermaid-${Math.random().toString(36).substr(2, 9)}`;
        
        try {
          const { svg } = await mermaid.render(id, chart);
          containerRef.current.innerHTML = svg;
        } catch (error) {
          console.error('Mermaid rendering error:', error);
          containerRef.current.innerHTML = `<div class="text-red-500 p-4">Error rendering diagram</div>`;
        }
      }
    };

    renderDiagram();
  }, [chart]);

  return (
    <div className="my-6">
      {title && (
        <h4 className="text-sm font-semibold mb-3" style={{ color: '#3b82f6' }}>
          {title}
        </h4>
      )}
      <div 
        className="p-6 rounded-lg overflow-x-auto"
        style={{ 
          background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(2, 6, 23, 0.8))',
          border: '1px solid rgba(59, 130, 246, 0.2)'
        }}
      >
        <div ref={containerRef} className="flex justify-center" />
      </div>
    </div>
  );
}
