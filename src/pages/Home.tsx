import { Link } from 'react-router-dom';
import { 
  Database, Sparkles, TreeDeciduous, Brain, 
  Link as LinkIcon, Boxes, AlertTriangle, Code, FileText, BookOpen,
  ArrowRight, Zap, Target, TrendingUp
} from 'lucide-react';

const modules = [
  {
    id: 1,
    title: 'Introduction to Data Mining',
    description: 'KDD process, data mining tasks, and real-world applications',
    icon: Database,
    color: '#3b82f6',
    topics: ['What is Data Mining?', 'KDD Process', 'Tasks & Applications'],
  },
  {
    id: 2,
    title: 'Data Preprocessing',
    description: 'Data cleaning, transformation, normalization, and dimensionality reduction',
    icon: Sparkles,
    color: '#8b5cf6',
    topics: ['Missing Values', 'Normalization', 'PCA'],
  },
  {
    id: 3,
    title: 'Classification Basics',
    description: 'Decision trees, entropy, information gain, and model evaluation',
    icon: TreeDeciduous,
    color: '#22c55e',
    topics: ['Decision Trees', 'Entropy', 'Confusion Matrix'],
  },
  {
    id: 4,
    title: 'Classification Advanced',
    description: 'k-NN, Naive Bayes, SVM, and ensemble methods',
    icon: Brain,
    color: '#f97316',
    topics: ['k-NN', 'SVM', 'Random Forest'],
  },
  {
    id: 5,
    title: 'Association Analysis',
    description: 'Market basket analysis, Apriori, and FP-Growth algorithms',
    icon: LinkIcon,
    color: '#06b6d4',
    topics: ['Support & Confidence', 'Apriori', 'FP-Growth'],
  },
  {
    id: 6,
    title: 'Cluster Analysis',
    description: 'K-Means, Hierarchical Clustering, and DBSCAN',
    icon: Boxes,
    color: '#a855f7',
    topics: ['K-Means', 'Hierarchical', 'DBSCAN'],
  },
  {
    id: 7,
    title: 'Anomaly Detection',
    description: 'Outlier detection methods and avoiding false discoveries',
    icon: AlertTriangle,
    color: '#ef4444',
    topics: ['LOF', 'Isolation Forest', 'False Discovery'],
  },
];

const features = [
  {
    title: 'Interactive Examples',
    description: 'Python code with scikit-learn, pandas, and numpy',
    icon: Code,
  },
  {
    title: 'Visual Diagrams',
    description: 'Algorithm flowcharts and concept visualizations',
    icon: TrendingUp,
  },
  {
    title: 'Quick Reference',
    description: 'Cheat sheets for formulas and algorithms',
    icon: FileText,
  },
  {
    title: 'Glossary',
    description: 'Comprehensive terminology reference',
    icon: BookOpen,
  },
];

export default function Home() {
  return (
    <div className="animate-fade-in">
      {/* Hero Section */}
      <section className="text-center mb-16">
        <div 
          className="inline-flex items-center gap-2 px-4 py-2 rounded-full mb-6"
          style={{ background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.3)' }}
        >
          <Zap size={16} style={{ color: '#3b82f6' }} />
          <span className="text-sm" style={{ color: '#3b82f6' }}>CPS844 - Winter 2026</span>
        </div>
        
        <h1 className="text-5xl font-bold mb-6">
          <span className="gradient-text">Data Mining</span>
          <br />
          <span className="text-white">Education Platform</span>
        </h1>
        
        <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-8">
          Master the fundamentals of extracting knowledge from data. From classification to clustering, 
          learn practical techniques with Python implementations.
        </p>

        <div className="flex justify-center gap-4">
          <Link
            to="/module/1"
            className="flex items-center gap-2 px-6 py-3 rounded-lg font-semibold text-white transition-all hover:scale-105"
            style={{ 
              background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
              boxShadow: '0 10px 30px rgba(59, 130, 246, 0.3)'
            }}
          >
            Start Learning
            <ArrowRight size={18} />
          </Link>
          <Link
            to="/playground"
            className="flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all hover:bg-white/10"
            style={{ 
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid rgba(59, 130, 246, 0.3)',
              color: '#3b82f6'
            }}
          >
            <Code size={18} />
            Open Playground
          </Link>
        </div>
      </section>

      {/* Stats */}
      <section className="grid grid-cols-4 gap-6 mb-16">
        {[
          { label: 'Modules', value: '7' },
          { label: 'Algorithms', value: '15+' },
          { label: 'Code Examples', value: '50+' },
          { label: 'Topics', value: '40+' },
        ].map((stat, index) => (
          <div 
            key={index}
            className="text-center p-6 rounded-lg"
            style={{ background: 'rgba(59, 130, 246, 0.05)', border: '1px solid rgba(59, 130, 246, 0.1)' }}
          >
            <div className="text-3xl font-bold gradient-text mb-1">{stat.value}</div>
            <div className="text-gray-400 text-sm">{stat.label}</div>
          </div>
        ))}
      </section>

      {/* Modules Grid */}
      <section className="mb-16">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-2xl font-bold text-white mb-2">Learning Modules</h2>
            <p className="text-gray-400">Complete curriculum from fundamentals to advanced topics</p>
          </div>
          <Link 
            to="/module/1"
            className="text-sm flex items-center gap-1 hover:text-white"
            style={{ color: '#3b82f6' }}
          >
            View All <ArrowRight size={14} />
          </Link>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {modules.map((module) => {
            const Icon = module.icon;
            return (
              <Link
                key={module.id}
                to={`/module/${module.id}`}
                className="card group hover:scale-[1.02] transition-all duration-300"
              >
                <div className="flex items-start gap-4">
                  <div 
                    className="p-3 rounded-lg"
                    style={{ background: `${module.color}20` }}
                  >
                    <Icon size={24} style={{ color: module.color }} />
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs font-medium px-2 py-0.5 rounded" style={{ background: `${module.color}20`, color: module.color }}>
                        Module {module.id}
                      </span>
                    </div>
                    <h3 className="font-semibold text-white group-hover:text-blue-400 transition-colors">
                      {module.title}
                    </h3>
                    <p className="text-sm text-gray-400 mt-1">{module.description}</p>
                    <div className="flex flex-wrap gap-2 mt-3">
                      {module.topics.map((topic, i) => (
                        <span 
                          key={i}
                          className="text-xs px-2 py-1 rounded"
                          style={{ background: 'rgba(255,255,255,0.05)', color: '#94a3b8' }}
                        >
                          {topic}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </Link>
            );
          })}
        </div>
      </section>

      {/* Features */}
      <section className="mb-16">
        <h2 className="text-2xl font-bold text-white mb-8 text-center">Platform Features</h2>
        <div className="grid md:grid-cols-4 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <div key={index} className="text-center p-6">
                <div 
                  className="w-12 h-12 rounded-lg flex items-center justify-center mx-auto mb-4"
                  style={{ background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.2))' }}
                >
                  <Icon size={24} style={{ color: '#3b82f6' }} />
                </div>
                <h3 className="font-semibold text-white mb-2">{feature.title}</h3>
                <p className="text-sm text-gray-400">{feature.description}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* Learning Path */}
      <section className="mb-16">
        <div 
          className="p-8 rounded-2xl"
          style={{ 
            background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1))',
            border: '1px solid rgba(59, 130, 246, 0.2)'
          }}
        >
          <div className="flex items-center gap-4 mb-6">
            <Target size={32} style={{ color: '#3b82f6' }} />
            <div>
              <h2 className="text-xl font-bold text-white">Recommended Learning Path</h2>
              <p className="text-gray-400">Follow this sequence for best results</p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-4">
            {modules.map((module, index) => (
              <div key={module.id} className="flex items-center gap-3">
                <Link
                  to={`/module/${module.id}`}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-white/5 transition-colors"
                  style={{ border: '1px solid rgba(255,255,255,0.1)' }}
                >
                  <span 
                    className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold"
                    style={{ background: module.color, color: 'white' }}
                  >
                    {module.id}
                  </span>
                  <span className="text-sm text-gray-300">{module.title.split(' ').slice(0, 2).join(' ')}</span>
                </Link>
                {index < modules.length - 1 && (
                  <ArrowRight size={16} className="text-gray-600" />
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="text-center">
        <div 
          className="p-12 rounded-2xl"
          style={{ 
            background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
            boxShadow: '0 20px 60px rgba(59, 130, 246, 0.3)'
          }}
        >
          <h2 className="text-3xl font-bold text-white mb-4">Ready to Start Mining?</h2>
          <p className="text-blue-100 mb-8 max-w-xl mx-auto">
            Begin your journey into data mining with comprehensive tutorials, 
            practical examples, and hands-on exercises.
          </p>
          <Link
            to="/module/1"
            className="inline-flex items-center gap-2 px-8 py-4 rounded-lg font-semibold text-blue-600 transition-all hover:scale-105"
            style={{ background: 'white' }}
          >
            Start Module 1
            <ArrowRight size={18} />
          </Link>
        </div>
      </section>
    </div>
  );
}
