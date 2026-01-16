import { Link } from 'react-router-dom';
import { Database, TrendingUp, Target, AlertTriangle, Globe, ShieldCheck } from 'lucide-react';
import CodeBlock from '../../components/code/CodeBlock';
import FlowDiagram from '../../components/visualizations/FlowDiagram';

const kddProcessDiagram = `
flowchart LR
    A[Raw Data] --> B[Selection]
    B --> C[Preprocessing]
    C --> D[Transformation]
    D --> E[Data Mining]
    E --> F[Evaluation]
    F --> G[Knowledge]
    
    style A fill:#1e293b,stroke:#3b82f6
    style B fill:#1e293b,stroke:#8b5cf6
    style C fill:#1e293b,stroke:#8b5cf6
    style D fill:#1e293b,stroke:#8b5cf6
    style E fill:#1e293b,stroke:#f97316
    style F fill:#1e293b,stroke:#22c55e
    style G fill:#1e293b,stroke:#22c55e
`;

const dataMiningTasksDiagram = `
flowchart TB
    DM[Data Mining Tasks]
    DM --> P[Predictive]
    DM --> D[Descriptive]
    
    P --> C[Classification]
    P --> R[Regression]
    
    D --> CL[Clustering]
    D --> AR[Association Rules]
    D --> AD[Anomaly Detection]
    
    style DM fill:#1e293b,stroke:#3b82f6
    style P fill:#1e293b,stroke:#f97316
    style D fill:#1e293b,stroke:#8b5cf6
    style C fill:#1e293b,stroke:#22c55e
    style R fill:#1e293b,stroke:#22c55e
    style CL fill:#1e293b,stroke:#06b6d4
    style AR fill:#1e293b,stroke:#06b6d4
    style AD fill:#1e293b,stroke:#06b6d4
`;

const simpleExample = `# Simple Data Mining Example: Pattern Discovery
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Load a classic dataset
iris = load_iris()
X, y = iris.data, iris.target

# Create and train a simple classifier
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

# The model has "mined" patterns from the data
print("Feature importances (discovered patterns):")
for name, importance in zip(iris.feature_names, clf.feature_importances_):
    print(f"  {name}: {importance:.3f}")

# Make predictions on new data
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = clf.predict(sample)
print(f"\\nPredicted class: {iris.target_names[prediction[0]]}")`;

export default function Module1_Introduction() {
  return (
    <div className="max-w-4xl mx-auto animate-fade-in">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-4">
          <div 
            className="p-3 rounded-lg"
            style={{ background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.2))' }}
          >
            <Database size={28} style={{ color: '#3b82f6' }} />
          </div>
          <div>
            <p className="text-sm font-medium" style={{ color: '#3b82f6' }}>Module 1</p>
            <h1 className="text-3xl font-bold text-white">Introduction to Data Mining</h1>
          </div>
        </div>
        <p className="text-gray-400 text-lg">
          Discover the fundamentals of extracting valuable knowledge from large datasets through 
          automated analysis techniques.
        </p>
      </div>

      {/* What is Data Mining */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <span className="text-2xl">💎</span> What is Data Mining?
        </h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            <strong className="text-white">Data Mining</strong> is the process of discovering patterns, 
            correlations, anomalies, and statistical relationships within large datasets using 
            machine learning, statistics, and database systems.
          </p>
          <div 
            className="p-4 rounded-lg mb-4"
            style={{ background: 'rgba(59, 130, 246, 0.1)', borderLeft: '4px solid #3b82f6' }}
          >
            <p className="text-gray-300 italic">
              "Data mining is the extraction of implicit, previously unknown, and potentially useful 
              information from data." — Witten & Frank
            </p>
          </div>
          <p className="text-gray-300">
            Unlike simple database queries that retrieve known information, data mining discovers 
            <span className="text-blue-400"> hidden patterns</span> and 
            <span className="text-purple-400"> predictive insights</span> that weren't explicitly 
            programmed or anticipated.
          </p>
        </div>
      </section>

      {/* KDD Process */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <span className="text-2xl">🔄</span> Knowledge Discovery in Databases (KDD)
        </h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            Data mining is a crucial step in the larger <strong className="text-white">KDD process</strong>, 
            which transforms raw data into actionable knowledge.
          </p>
          
          <FlowDiagram chart={kddProcessDiagram} title="The KDD Process Pipeline" />

          <div className="grid md:grid-cols-2 gap-4 mt-6">
            <div className="p-4 rounded-lg" style={{ background: 'rgba(139, 92, 246, 0.1)' }}>
              <h4 className="font-semibold text-purple-400 mb-2">1. Selection</h4>
              <p className="text-gray-400 text-sm">
                Identify and select relevant data from the target database for analysis.
              </p>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(139, 92, 246, 0.1)' }}>
              <h4 className="font-semibold text-purple-400 mb-2">2. Preprocessing</h4>
              <p className="text-gray-400 text-sm">
                Clean data by handling missing values, removing noise, and resolving inconsistencies.
              </p>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(139, 92, 246, 0.1)' }}>
              <h4 className="font-semibold text-purple-400 mb-2">3. Transformation</h4>
              <p className="text-gray-400 text-sm">
                Convert data into appropriate formats (normalization, aggregation, feature engineering).
              </p>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(249, 115, 22, 0.1)' }}>
              <h4 className="font-semibold text-orange-400 mb-2">4. Data Mining</h4>
              <p className="text-gray-400 text-sm">
                Apply algorithms to extract patterns (classification, clustering, association rules).
              </p>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(34, 197, 94, 0.1)' }}>
              <h4 className="font-semibold text-green-400 mb-2">5. Evaluation</h4>
              <p className="text-gray-400 text-sm">
                Assess discovered patterns for validity, novelty, usefulness, and understandability.
              </p>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(34, 197, 94, 0.1)' }}>
              <h4 className="font-semibold text-green-400 mb-2">6. Knowledge</h4>
              <p className="text-gray-400 text-sm">
                Present and integrate discovered knowledge into decision-making systems.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Data Mining Tasks */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <Target size={28} style={{ color: '#f97316' }} /> Data Mining Tasks
        </h2>
        <div className="card">
          <FlowDiagram chart={dataMiningTasksDiagram} title="Taxonomy of Data Mining Tasks" />

          <div className="mt-6 space-y-4">
            <div className="p-4 rounded-lg" style={{ background: 'rgba(249, 115, 22, 0.1)', border: '1px solid rgba(249, 115, 22, 0.3)' }}>
              <h4 className="font-bold text-orange-400 mb-2 flex items-center gap-2">
                <TrendingUp size={18} /> Predictive Tasks
              </h4>
              <ul className="text-gray-300 space-y-2 ml-6">
                <li>
                  <strong className="text-white">Classification:</strong> Predict categorical labels 
                  (e.g., spam/not spam, fraud/legitimate)
                </li>
                <li>
                  <strong className="text-white">Regression:</strong> Predict continuous values 
                  (e.g., stock prices, temperature)
                </li>
              </ul>
            </div>

            <div className="p-4 rounded-lg" style={{ background: 'rgba(139, 92, 246, 0.1)', border: '1px solid rgba(139, 92, 246, 0.3)' }}>
              <h4 className="font-bold text-purple-400 mb-2 flex items-center gap-2">
                <Database size={18} /> Descriptive Tasks
              </h4>
              <ul className="text-gray-300 space-y-2 ml-6">
                <li>
                  <strong className="text-white">Clustering:</strong> Group similar data points 
                  (e.g., customer segmentation)
                </li>
                <li>
                  <strong className="text-white">Association Rules:</strong> Find relationships 
                  (e.g., "customers who buy X also buy Y")
                </li>
                <li>
                  <strong className="text-white">Anomaly Detection:</strong> Identify unusual patterns 
                  (e.g., fraud detection, system failures)
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Applications */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <Globe size={28} style={{ color: '#06b6d4' }} /> Real-World Applications
        </h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="card">
            <h4 className="font-bold text-cyan-400 mb-3">🌐 Web Mining</h4>
            <ul className="text-gray-400 space-y-2 text-sm">
              <li>• Search engine ranking (PageRank)</li>
              <li>• Click-stream analysis</li>
              <li>• Content recommendation</li>
              <li>• Social network analysis</li>
            </ul>
          </div>
          <div className="card">
            <h4 className="font-bold text-blue-400 mb-3">🛒 E-Commerce</h4>
            <ul className="text-gray-400 space-y-2 text-sm">
              <li>• Product recommendations</li>
              <li>• Market basket analysis</li>
              <li>• Customer churn prediction</li>
              <li>• Dynamic pricing</li>
            </ul>
          </div>
          <div className="card">
            <h4 className="font-bold text-green-400 mb-3">🏥 Healthcare</h4>
            <ul className="text-gray-400 space-y-2 text-sm">
              <li>• Disease diagnosis</li>
              <li>• Drug discovery</li>
              <li>• Patient outcome prediction</li>
              <li>• Medical image analysis</li>
            </ul>
          </div>
          <div className="card">
            <h4 className="font-bold text-red-400 mb-3">🔒 Security</h4>
            <ul className="text-gray-400 space-y-2 text-sm">
              <li>• Fraud detection</li>
              <li>• Intrusion detection</li>
              <li>• Spam filtering</li>
              <li>• Identity verification</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Code Example */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <span className="text-2xl">💻</span> Python Example
        </h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            Here's a simple example showing how data mining algorithms can discover patterns in data:
          </p>
          <CodeBlock code={simpleExample} language="python" title="intro_example.py" />
        </div>
      </section>

      {/* Challenges */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <AlertTriangle size={28} style={{ color: '#eab308' }} /> Challenges in Data Mining
        </h2>
        <div className="card">
          <div className="grid md:grid-cols-2 gap-4">
            <div className="p-4 rounded-lg" style={{ background: 'rgba(234, 179, 8, 0.1)' }}>
              <h4 className="font-semibold text-yellow-400 mb-2">Scalability</h4>
              <p className="text-gray-400 text-sm">
                Handling massive datasets (Big Data) with billions of records and thousands of features.
              </p>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(234, 179, 8, 0.1)' }}>
              <h4 className="font-semibold text-yellow-400 mb-2">High Dimensionality</h4>
              <p className="text-gray-400 text-sm">
                The "curse of dimensionality" - many features make pattern discovery harder.
              </p>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(234, 179, 8, 0.1)' }}>
              <h4 className="font-semibold text-yellow-400 mb-2">Data Quality</h4>
              <p className="text-gray-400 text-sm">
                Missing values, noise, inconsistencies, and duplicates affect results.
              </p>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(234, 179, 8, 0.1)' }}>
              <h4 className="font-semibold text-yellow-400 mb-2">Privacy & Ethics</h4>
              <p className="text-gray-400 text-sm">
                Protecting sensitive information while extracting useful patterns.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Ethics Section */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <ShieldCheck size={28} style={{ color: '#22c55e' }} /> Ethical Considerations
        </h2>
        <div className="card">
          <div 
            className="p-4 rounded-lg mb-4"
            style={{ background: 'rgba(34, 197, 94, 0.1)', borderLeft: '4px solid #22c55e' }}
          >
            <p className="text-gray-300">
              Data mining raises important ethical questions about privacy, fairness, and transparency. 
              Practitioners must consider the societal impact of their models.
            </p>
          </div>
          <ul className="text-gray-300 space-y-3">
            <li className="flex items-start gap-2">
              <span className="text-green-400">✓</span>
              <span><strong className="text-white">Privacy:</strong> Ensure data collection and analysis comply with regulations (GDPR, HIPAA)</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-400">✓</span>
              <span><strong className="text-white">Fairness:</strong> Avoid discriminatory biases in predictive models</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-400">✓</span>
              <span><strong className="text-white">Transparency:</strong> Make model decisions explainable and interpretable</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-400">✓</span>
              <span><strong className="text-white">Consent:</strong> Obtain proper authorization before collecting personal data</span>
            </li>
          </ul>
        </div>
      </section>

      {/* Key Takeaways */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4">📌 Key Takeaways</h2>
        <div 
          className="p-6 rounded-lg"
          style={{ 
            background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1))',
            border: '1px solid rgba(59, 130, 246, 0.3)'
          }}
        >
          <ul className="space-y-3 text-gray-300">
            <li className="flex items-start gap-3">
              <span className="text-blue-400 font-bold">1.</span>
              <span>Data mining extracts <strong className="text-white">hidden patterns</strong> from large datasets</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-blue-400 font-bold">2.</span>
              <span>It's part of the larger <strong className="text-white">KDD process</strong> (Selection → Preprocessing → Transformation → Mining → Evaluation)</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-blue-400 font-bold">3.</span>
              <span>Tasks include <strong className="text-white">classification, clustering, association rules, and anomaly detection</strong></span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-blue-400 font-bold">4.</span>
              <span>Applications span <strong className="text-white">web mining, healthcare, finance, and security</strong></span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-blue-400 font-bold">5.</span>
              <span>Always consider <strong className="text-white">ethical implications</strong> and data privacy</span>
            </li>
          </ul>
        </div>
      </section>

      {/* Navigation */}
      <div className="flex justify-between items-center pt-8 border-t" style={{ borderColor: 'rgba(59, 130, 246, 0.2)' }}>
        <div />
        <Link
          to="/module/2"
          className="btn-primary flex items-center gap-2"
        >
          Next: Data Preprocessing
          <span>→</span>
        </Link>
      </div>
    </div>
  );
}
