import { FileText, Copy, Check } from 'lucide-react';
import { useState } from 'react';

const sections = [
  {
    title: 'Preprocessing Formulas',
    items: [
      { name: 'Min-Max Normalization', formula: "X' = (X - X_min) / (X_max - X_min)", description: 'Scales to [0, 1]' },
      { name: 'Z-Score Standardization', formula: "X' = (X - μ) / σ", description: 'Mean=0, Std=1' },
      { name: 'Missing Value (Mean)', formula: 'X_missing = mean(X)', description: 'Replace with column mean' },
    ],
  },
  {
    title: 'Classification Metrics',
    items: [
      { name: 'Accuracy', formula: '(TP + TN) / (TP + TN + FP + FN)', description: 'Overall correctness' },
      { name: 'Precision', formula: 'TP / (TP + FP)', description: 'Positive predictive value' },
      { name: 'Recall', formula: 'TP / (TP + FN)', description: 'Sensitivity, True Positive Rate' },
      { name: 'F1 Score', formula: '2 × (P × R) / (P + R)', description: 'Harmonic mean of P and R' },
      { name: 'Specificity', formula: 'TN / (TN + FP)', description: 'True Negative Rate' },
    ],
  },
  {
    title: 'Decision Tree Metrics',
    items: [
      { name: 'Entropy', formula: 'H(S) = -Σ pᵢ log₂(pᵢ)', description: 'Impurity measure (0-1)' },
      { name: 'Information Gain', formula: 'IG = H(parent) - Σ(|Sᵥ|/|S|)H(Sᵥ)', description: 'Entropy reduction' },
      { name: 'Gini Impurity', formula: 'Gini = 1 - Σ pᵢ²', description: 'Used in CART' },
    ],
  },
  {
    title: 'Association Rules',
    items: [
      { name: 'Support', formula: 'Support(X) = count(X) / N', description: 'Frequency of itemset' },
      { name: 'Confidence', formula: 'Conf(X→Y) = Support(X∪Y) / Support(X)', description: 'Rule strength' },
      { name: 'Lift', formula: 'Lift(X→Y) = Conf(X→Y) / Support(Y)', description: '>1 positive association' },
    ],
  },
  {
    title: 'Clustering Metrics',
    items: [
      { name: 'Inertia (SSE)', formula: 'Σ ||xᵢ - μₖ||²', description: 'Within-cluster sum of squares' },
      { name: 'Silhouette Score', formula: '(b - a) / max(a, b)', description: 'Range: -1 to 1, higher is better' },
      { name: 'Euclidean Distance', formula: '√Σ(xᵢ - yᵢ)²', description: 'L2 distance' },
      { name: 'Manhattan Distance', formula: 'Σ|xᵢ - yᵢ|', description: 'L1 distance' },
    ],
  },
  {
    title: 'Anomaly Detection',
    items: [
      { name: 'Z-Score Outlier', formula: '|z| > 3 is outlier', description: '3 standard deviations' },
      { name: 'IQR Outlier', formula: 'x < Q1-1.5×IQR or x > Q3+1.5×IQR', description: 'Interquartile method' },
      { name: 'LOF Score', formula: 'LOF > 1 indicates anomaly', description: 'Local density comparison' },
    ],
  },
];

const algorithms = [
  {
    category: 'Classification',
    items: [
      { name: 'Decision Tree', complexity: 'O(n × m × log n)', pros: 'Interpretable', cons: 'Overfitting' },
      { name: 'k-NN', complexity: 'O(n × d)', pros: 'Simple, no training', cons: 'Slow prediction' },
      { name: 'Naive Bayes', complexity: 'O(n × m)', pros: 'Fast, handles missing', cons: 'Independence assumption' },
      { name: 'SVM', complexity: 'O(n² to n³)', pros: 'Effective high-dim', cons: 'Slow large datasets' },
      { name: 'Random Forest', complexity: 'O(k × n × m × log n)', pros: 'Robust, feature importance', cons: 'Less interpretable' },
    ],
  },
  {
    category: 'Clustering',
    items: [
      { name: 'K-Means', complexity: 'O(n × k × t)', pros: 'Fast, scalable', cons: 'Must specify K' },
      { name: 'Hierarchical', complexity: 'O(n² log n)', pros: 'Dendrogram, no K', cons: 'Slow large data' },
      { name: 'DBSCAN', complexity: 'O(n log n)', pros: 'Handles noise, shapes', cons: 'eps sensitivity' },
    ],
  },
  {
    category: 'Association',
    items: [
      { name: 'Apriori', complexity: 'O(2^m)', pros: 'Simple, interpretable', cons: 'Multiple DB scans' },
      { name: 'FP-Growth', complexity: 'O(n)', pros: 'Fast, 2 scans', cons: 'FP-tree memory' },
    ],
  },
];

const pythonSnippets = [
  {
    title: 'Quick Imports',
    code: `import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler`,
  },
  {
    title: 'Train-Test Split',
    code: `X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)`,
  },
  {
    title: 'Classification Pipeline',
    code: `from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)`,
  },
  {
    title: 'K-Means Clustering',
    code: `from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)`,
  },
  {
    title: 'Cross-Validation',
    code: `from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, y, cv=5)
print(f"CV Accuracy: {scores.mean():.3f}")`,
  },
];

export default function CheatSheet() {
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const copySnippet = async (code: string, index: number) => {
    await navigator.clipboard.writeText(code);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  return (
    <div className="max-w-6xl mx-auto animate-fade-in">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-4">
          <div 
            className="p-3 rounded-lg"
            style={{ background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.2))' }}
          >
            <FileText size={28} style={{ color: '#3b82f6' }} />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-white">Cheat Sheet</h1>
            <p className="text-gray-400">Quick reference for formulas, metrics, and code snippets</p>
          </div>
        </div>
      </div>

      {/* Formulas Section */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-6">📐 Key Formulas</h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {sections.map((section, sIdx) => (
            <div 
              key={sIdx}
              className="p-5 rounded-lg"
              style={{ background: 'rgba(15, 23, 42, 0.8)', border: '1px solid rgba(59, 130, 246, 0.2)' }}
            >
              <h3 className="font-semibold text-blue-400 mb-4">{section.title}</h3>
              <div className="space-y-3">
                {section.items.map((item, iIdx) => (
                  <div key={iIdx} className="border-b border-gray-800 pb-3 last:border-0 last:pb-0">
                    <div className="font-medium text-white text-sm">{item.name}</div>
                    <code className="text-cyan-400 text-xs block mt-1">{item.formula}</code>
                    <p className="text-gray-500 text-xs mt-1">{item.description}</p>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Algorithms Comparison */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-6">🔬 Algorithm Comparison</h2>
        {algorithms.map((cat, cIdx) => (
          <div key={cIdx} className="mb-6">
            <h3 className="text-lg font-semibold text-purple-400 mb-3">{cat.category}</h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-left">
                    <th className="py-2 px-4">Algorithm</th>
                    <th className="py-2 px-4">Complexity</th>
                    <th className="py-2 px-4">Pros</th>
                    <th className="py-2 px-4">Cons</th>
                  </tr>
                </thead>
                <tbody>
                  {cat.items.map((item, iIdx) => (
                    <tr key={iIdx} className="border-t border-gray-800">
                      <td className="py-2 px-4 font-medium text-white">{item.name}</td>
                      <td className="py-2 px-4 text-cyan-400 font-mono text-sm">{item.complexity}</td>
                      <td className="py-2 px-4 text-green-400 text-sm">{item.pros}</td>
                      <td className="py-2 px-4 text-red-400 text-sm">{item.cons}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ))}
      </section>

      {/* Python Snippets */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-6">🐍 Python Snippets</h2>
        <div className="grid md:grid-cols-2 gap-4">
          {pythonSnippets.map((snippet, idx) => (
            <div 
              key={idx}
              className="rounded-lg overflow-hidden"
              style={{ background: 'linear-gradient(135deg, #0c1220, #0a0f1a)', border: '1px solid rgba(59, 130, 246, 0.3)' }}
            >
              <div 
                className="px-4 py-2 flex items-center justify-between"
                style={{ background: 'rgba(59, 130, 246, 0.1)', borderBottom: '1px solid rgba(59, 130, 246, 0.2)' }}
              >
                <span className="text-sm font-medium text-gray-300">{snippet.title}</span>
                <button
                  onClick={() => copySnippet(snippet.code, idx)}
                  className="p-1 rounded hover:bg-white/10 transition-colors"
                >
                  {copiedIndex === idx ? (
                    <Check size={14} className="text-green-400" />
                  ) : (
                    <Copy size={14} className="text-gray-400" />
                  )}
                </button>
              </div>
              <pre className="p-4 text-sm font-mono text-gray-300 overflow-x-auto">
                <code>{snippet.code}</code>
              </pre>
            </div>
          ))}
        </div>
      </section>

      {/* Quick Reference */}
      <section>
        <h2 className="text-2xl font-bold text-white mb-6">⚡ Quick Reference</h2>
        <div className="grid md:grid-cols-3 gap-6">
          <div 
            className="p-5 rounded-lg"
            style={{ background: 'rgba(34, 197, 94, 0.1)', border: '1px solid rgba(34, 197, 94, 0.3)' }}
          >
            <h3 className="font-semibold text-green-400 mb-3">When to Use What</h3>
            <ul className="text-sm text-gray-300 space-y-2">
              <li><strong>Classification:</strong> Predict categories</li>
              <li><strong>Regression:</strong> Predict continuous</li>
              <li><strong>Clustering:</strong> Find groups</li>
              <li><strong>Association:</strong> Find relationships</li>
              <li><strong>Anomaly:</strong> Find outliers</li>
            </ul>
          </div>
          <div 
            className="p-5 rounded-lg"
            style={{ background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.3)' }}
          >
            <h3 className="font-semibold text-blue-400 mb-3">Preprocessing Steps</h3>
            <ol className="text-sm text-gray-300 space-y-2 list-decimal list-inside">
              <li>Handle missing values</li>
              <li>Encode categories</li>
              <li>Scale/normalize features</li>
              <li>Remove outliers</li>
              <li>Feature selection</li>
            </ol>
          </div>
          <div 
            className="p-5 rounded-lg"
            style={{ background: 'rgba(249, 115, 22, 0.1)', border: '1px solid rgba(249, 115, 22, 0.3)' }}
          >
            <h3 className="font-semibold text-orange-400 mb-3">Common Mistakes</h3>
            <ul className="text-sm text-gray-300 space-y-2">
              <li>❌ Data leakage in scaling</li>
              <li>❌ Not handling imbalanced classes</li>
              <li>❌ Ignoring feature scaling for k-NN</li>
              <li>❌ Overfitting without validation</li>
              <li>❌ Using accuracy on imbalanced data</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}
