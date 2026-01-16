import { Link, useLocation } from 'react-router-dom';
import {
  Home,
  Database,
  Sparkles,
  TreeDeciduous,
  Brain,
  Link as LinkIcon,
  Boxes,
  AlertTriangle,
  Code,
  FileText,
  BookOpen,
  Linkedin,
} from 'lucide-react';

const navigation = [
  { name: 'Home', href: '/', icon: Home },
  { name: 'Module 1: Introduction', href: '/module/1', icon: Database },
  { name: 'Module 2: Data Preprocessing', href: '/module/2', icon: Sparkles },
  { name: 'Module 3: Classification Basics', href: '/module/3', icon: TreeDeciduous },
  { name: 'Module 4: Classification Advanced', href: '/module/4', icon: Brain },
  { name: 'Module 5: Association Analysis', href: '/module/5', icon: LinkIcon },
  { name: 'Module 6: Cluster Analysis', href: '/module/6', icon: Boxes },
  { name: 'Module 7: Anomaly Detection', href: '/module/7', icon: AlertTriangle },
  { name: 'Playground', href: '/playground', icon: Code },
  { name: 'Cheat Sheet', href: '/cheatsheet', icon: FileText },
  { name: 'Glossary', href: '/glossary', icon: BookOpen },
];

export default function Sidebar() {
  const location = useLocation();

  return (
    <aside 
      style={{ 
        position: 'fixed',
        left: 0,
        top: 0,
        height: '100vh',
        width: '288px',
        display: 'flex',
        flexDirection: 'column',
        background: 'linear-gradient(180deg, #0c1629 0%, #0a1120 100%)',
        borderRight: '1px solid rgba(59, 130, 246, 0.2)',
        zIndex: 50,
      }}
    >
      {/* Logo */}
      <div className="p-6 border-b" style={{ borderColor: 'rgba(59, 130, 246, 0.2)' }}>
        <Link to="/" className="flex items-center gap-3">
          <div 
            className="w-10 h-10 rounded-lg flex items-center justify-center"
            style={{ 
              background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
              boxShadow: '0 0 20px rgba(59, 130, 246, 0.4)'
            }}
          >
            <Database size={24} className="text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white">Data Mining</h1>
            <p className="text-xs" style={{ color: '#64748b' }}>CPS844 Education</p>
          </div>
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 overflow-y-auto">
        <ul className="space-y-1">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href;
            const Icon = item.icon;
            
            return (
              <li key={item.name}>
                <Link
                  to={item.href}
                  className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                    isActive 
                      ? 'text-white' 
                      : 'text-gray-400 hover:text-white hover:bg-white/5'
                  }`}
                  style={isActive ? {
                    background: 'linear-gradient(90deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.1))',
                    borderLeft: '3px solid #3b82f6',
                  } : {}}
                >
                  <Icon size={18} style={{ color: isActive ? '#3b82f6' : undefined }} />
                  <span className="text-sm font-medium">{item.name}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* Footer */}
      <div className="p-4 border-t" style={{ borderColor: 'rgba(59, 130, 246, 0.2)' }}>
        {/* LinkedIn Button */}
        <a
          href="https://www.linkedin.com/in/donald-jung/"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center justify-center gap-2 mb-3 px-4 py-2 rounded-lg text-white transition-all hover:scale-105"
          style={{
            background: 'linear-gradient(45deg, #0077B5, #005582)',
            boxShadow: '0 4px 15px rgba(0, 119, 181, 0.4)',
          }}
        >
          <Linkedin size={18} />
          <span className="font-semibold text-sm">Connect on LinkedIn</span>
        </a>

        {/* Watermark */}
        <div className="text-center mt-4">
          <p className="text-xs text-gray-500 uppercase tracking-wider">crafted with ♥ by</p>
          <p 
            className="text-sm font-bold mt-1"
            style={{
              background: 'linear-gradient(45deg, #3b82f6, #8b5cf6, #f97316)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Donald Jung
          </p>
        </div>
        <p className="text-xs text-gray-500 text-center mt-2">Data Mining Platform v1.0</p>
      </div>
    </aside>
  );
}
