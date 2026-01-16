import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, X, Command } from 'lucide-react';

const searchData = [
  { title: 'Introduction to Data Mining', path: '/module/1', category: 'Module' },
  { title: 'KDD Process', path: '/module/1', category: 'Concept' },
  { title: 'Data Preprocessing', path: '/module/2', category: 'Module' },
  { title: 'Data Cleaning', path: '/module/2', category: 'Concept' },
  { title: 'Normalization', path: '/module/2', category: 'Concept' },
  { title: 'Classification Basics', path: '/module/3', category: 'Module' },
  { title: 'Decision Trees', path: '/module/3', category: 'Algorithm' },
  { title: 'Entropy', path: '/module/3', category: 'Concept' },
  { title: 'Information Gain', path: '/module/3', category: 'Concept' },
  { title: 'Classification Advanced', path: '/module/4', category: 'Module' },
  { title: 'K-Nearest Neighbors', path: '/module/4', category: 'Algorithm' },
  { title: 'Naive Bayes', path: '/module/4', category: 'Algorithm' },
  { title: 'Support Vector Machines', path: '/module/4', category: 'Algorithm' },
  { title: 'Random Forest', path: '/module/4', category: 'Algorithm' },
  { title: 'Association Analysis', path: '/module/5', category: 'Module' },
  { title: 'Apriori Algorithm', path: '/module/5', category: 'Algorithm' },
  { title: 'FP-Growth', path: '/module/5', category: 'Algorithm' },
  { title: 'Support and Confidence', path: '/module/5', category: 'Concept' },
  { title: 'Cluster Analysis', path: '/module/6', category: 'Module' },
  { title: 'K-Means Clustering', path: '/module/6', category: 'Algorithm' },
  { title: 'DBSCAN', path: '/module/6', category: 'Algorithm' },
  { title: 'Hierarchical Clustering', path: '/module/6', category: 'Algorithm' },
  { title: 'Anomaly Detection', path: '/module/7', category: 'Module' },
  { title: 'Isolation Forest', path: '/module/7', category: 'Algorithm' },
  { title: 'Local Outlier Factor', path: '/module/7', category: 'Algorithm' },
  { title: 'Playground', path: '/playground', category: 'Tool' },
  { title: 'Cheat Sheet', path: '/cheatsheet', category: 'Reference' },
  { title: 'Glossary', path: '/glossary', category: 'Reference' },
];

export default function Header() {
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const navigate = useNavigate();

  const filteredResults = searchData.filter(item =>
    item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    item.category.toLowerCase().includes(searchQuery.toLowerCase())
  );

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsSearchOpen(true);
      }
      if (e.key === 'Escape') {
        setIsSearchOpen(false);
        setSearchQuery('');
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  useEffect(() => {
    if (!isSearchOpen) {
      setSearchQuery('');
      setSelectedIndex(0);
    }
  }, [isSearchOpen]);

  useEffect(() => {
    setSelectedIndex(0);
  }, [searchQuery]);

  const handleKeyNavigation = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex(prev => Math.min(prev + 1, filteredResults.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex(prev => Math.max(prev - 1, 0));
    } else if (e.key === 'Enter' && filteredResults[selectedIndex]) {
      navigate(filteredResults[selectedIndex].path);
      setIsSearchOpen(false);
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'Module': return '#3b82f6';
      case 'Algorithm': return '#8b5cf6';
      case 'Concept': return '#f97316';
      case 'Tool': return '#22c55e';
      case 'Reference': return '#06b6d4';
      default: return '#64748b';
    }
  };

  return (
    <>
      <header 
        className="h-16 flex items-center justify-between px-8 border-b"
        style={{ 
          background: 'rgba(12, 22, 41, 0.8)',
          backdropFilter: 'blur(10px)',
          borderColor: 'rgba(59, 130, 246, 0.2)'
        }}
      >
        <div className="flex items-center gap-4">
          <h2 className="text-lg font-semibold text-white">Data Mining Education</h2>
          <span className="px-2 py-1 rounded text-xs font-medium" style={{ background: 'rgba(59, 130, 246, 0.2)', color: '#3b82f6' }}>
            CPS844
          </span>
        </div>

        <button
          onClick={() => setIsSearchOpen(true)}
          className="flex items-center gap-3 px-4 py-2 rounded-lg transition-all hover:bg-white/5"
          style={{ 
            background: 'rgba(255, 255, 255, 0.05)',
            border: '1px solid rgba(59, 130, 246, 0.2)'
          }}
        >
          <Search size={16} className="text-gray-400" />
          <span className="text-gray-400 text-sm">Search documentation...</span>
          <div className="flex items-center gap-1 ml-4">
            <kbd className="px-2 py-0.5 rounded text-xs bg-white/10 text-gray-400 flex items-center gap-1">
              <Command size={10} />K
            </kbd>
          </div>
        </button>
      </header>

      {/* Search Modal */}
      {isSearchOpen && (
        <div 
          className="fixed inset-0 z-50 flex items-start justify-center pt-24"
          style={{ background: 'rgba(0, 0, 0, 0.8)' }}
          onClick={() => setIsSearchOpen(false)}
        >
          <div 
            className="w-full max-w-2xl rounded-xl overflow-hidden shadow-2xl"
            style={{ 
              background: 'linear-gradient(135deg, #0c1629, #0a1120)',
              border: '1px solid rgba(59, 130, 246, 0.3)'
            }}
            onClick={e => e.stopPropagation()}
          >
            {/* Search Input */}
            <div className="flex items-center gap-3 p-4 border-b" style={{ borderColor: 'rgba(59, 130, 246, 0.2)' }}>
              <Search size={20} style={{ color: '#3b82f6' }} />
              <input
                type="text"
                placeholder="Search algorithms, concepts, modules..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={handleKeyNavigation}
                className="flex-1 bg-transparent text-white placeholder-gray-500 outline-none text-lg"
                autoFocus
              />
              <button onClick={() => setIsSearchOpen(false)} className="p-1 hover:bg-white/10 rounded">
                <X size={18} className="text-gray-400" />
              </button>
            </div>

            {/* Results */}
            <div className="max-h-96 overflow-y-auto p-2">
              {filteredResults.length === 0 ? (
                <div className="p-8 text-center text-gray-500">
                  No results found for "{searchQuery}"
                </div>
              ) : (
                filteredResults.map((item, index) => (
                  <button
                    key={`${item.path}-${item.title}`}
                    onClick={() => {
                      navigate(item.path);
                      setIsSearchOpen(false);
                    }}
                    className={`w-full flex items-center gap-3 p-3 rounded-lg text-left transition-all ${
                      index === selectedIndex ? 'bg-white/10' : 'hover:bg-white/5'
                    }`}
                  >
                    <span 
                      className="px-2 py-1 rounded text-xs font-medium"
                      style={{ 
                        background: `${getCategoryColor(item.category)}20`,
                        color: getCategoryColor(item.category)
                      }}
                    >
                      {item.category}
                    </span>
                    <span className="text-white">{item.title}</span>
                  </button>
                ))
              )}
            </div>

            {/* Footer */}
            <div className="flex items-center gap-4 p-3 border-t text-xs text-gray-500" style={{ borderColor: 'rgba(59, 130, 246, 0.2)' }}>
              <span className="flex items-center gap-1"><kbd className="px-1.5 py-0.5 rounded bg-white/10">↑↓</kbd> Navigate</span>
              <span className="flex items-center gap-1"><kbd className="px-1.5 py-0.5 rounded bg-white/10">↵</kbd> Select</span>
              <span className="flex items-center gap-1"><kbd className="px-1.5 py-0.5 rounded bg-white/10">Esc</kbd> Close</span>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
