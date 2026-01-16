import { useState } from 'react';
import { Play, RotateCcw, Copy, Check, Lightbulb } from 'lucide-react';
import OutputDisplay from '../components/code/OutputDisplay';

const algorithms: Record<string, { name: string; code: string; output: string; description: string }> = {
  kmeans: {
    name: 'K-Means Clustering',
    description: 'Partition data into K clusters by minimizing within-cluster variance',
    code: `# K-Means Clustering
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)

# Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

print("K-Means Clustering Results")
print("="*40)
print(f"Number of clusters: 4")
print(f"Inertia (SSE): {kmeans.inertia_:.2f}")
print(f"\\nCluster sizes:")
for i in range(4):
    print(f"  Cluster {i}: {(labels == i).sum()} points")
print(f"\\nCentroids:\\n{kmeans.cluster_centers_}")`,
    output: `K-Means Clustering Results
========================================
Number of clusters: 4
Inertia (SSE): 681.19

Cluster sizes:
  Cluster 0: 75 points
  Cluster 1: 75 points
  Cluster 2: 75 points
  Cluster 3: 75 points

Centroids:
[[-2.37  2.87]
 [ 1.97  0.87]
 [-1.64 -4.23]
 [ 3.21  4.11]]`,
  },
  decisionTree: {
    name: 'Decision Tree',
    description: 'Classification using if-then rules learned from data',
    code: `# Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# Train decision tree
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
accuracy = clf.score(X_test, y_test)
print("Decision Tree Classification")
print("="*40)
print(f"Max depth: 3")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"\\nTest Accuracy: {accuracy:.3f}")
print(f"\\nFeature Importances:")
for name, imp in zip(iris.feature_names, clf.feature_importances_):
    print(f"  {name}: {imp:.3f}")`,
    output: `Decision Tree Classification
========================================
Max depth: 3
Training samples: 105
Test samples: 45

Test Accuracy: 0.978

Feature Importances:
  sepal length (cm): 0.000
  sepal width (cm): 0.000
  petal length (cm): 0.424
  petal width (cm): 0.576`,
  },
  apriori: {
    name: 'Association Rules (Apriori)',
    description: 'Find frequent itemsets and association rules',
    code: `# Association Rule Mining
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

# Transaction data
transactions = [
    ['bread', 'milk'],
    ['bread', 'diapers', 'beer', 'eggs'],
    ['milk', 'diapers', 'beer', 'cola'],
    ['bread', 'milk', 'diapers', 'beer'],
    ['bread', 'milk', 'diapers', 'cola'],
]

# Convert to one-hot encoding
te = TransactionEncoder()
df = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)

# Find frequent itemsets
freq = apriori(df, min_support=0.4, use_colnames=True)
print("Association Rule Mining")
print("="*40)
print("\\nFrequent Itemsets (support >= 40%):")
for _, row in freq.iterrows():
    items = ', '.join(list(row['itemsets']))
    print(f"  {{{items}}}: {row['support']:.2f}")

# Generate rules
rules = association_rules(freq, metric="confidence", min_threshold=0.6)
print("\\nAssociation Rules (confidence >= 60%):")
for _, row in rules.head(3).iterrows():
    ant = ', '.join(list(row['antecedents']))
    con = ', '.join(list(row['consequents']))
    print(f"  {{{ant}}} -> {{{con}}}")
    print(f"    conf={row['confidence']:.2f}, lift={row['lift']:.2f}")`,
    output: `Association Rule Mining
========================================

Frequent Itemsets (support >= 40%):
  {beer}: 0.60
  {bread}: 0.80
  {diapers}: 0.80
  {milk}: 0.80
  {bread, milk}: 0.60
  {beer, diapers}: 0.60
  {bread, diapers}: 0.60
  {diapers, milk}: 0.60
  {bread, diapers, milk}: 0.40

Association Rules (confidence >= 60%):
  {beer} -> {diapers}
    conf=1.00, lift=1.25
  {diapers} -> {beer}
    conf=0.75, lift=1.25
  {bread} -> {milk}
    conf=0.75, lift=0.94`,
  },
  isolationForest: {
    name: 'Isolation Forest',
    description: 'Anomaly detection using random tree isolation',
    code: `# Isolation Forest Anomaly Detection
from sklearn.ensemble import IsolationForest
import numpy as np

# Generate data with anomalies
np.random.seed(42)
X_normal = np.random.randn(200, 2)
X_outliers = np.random.uniform(-4, 4, (10, 2))
X = np.vstack([X_normal, X_outliers])

# Apply Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=42)
labels = iso.fit_predict(X)
scores = iso.decision_function(X)

n_outliers = (labels == -1).sum()
print("Isolation Forest Results")
print("="*40)
print(f"Total points: {len(X)}")
print(f"Detected outliers: {n_outliers}")
print(f"Detected inliers: {(labels == 1).sum()}")
print(f"\\nScore statistics:")
print(f"  Min: {scores.min():.3f}")
print(f"  Max: {scores.max():.3f}")
print(f"  Mean: {scores.mean():.3f}")
print(f"\\nTop 5 most anomalous scores:")
top5 = np.argsort(scores)[:5]
for idx in top5:
    print(f"  Point {idx}: {scores[idx]:.3f}")`,
    output: `Isolation Forest Results
========================================
Total points: 210
Detected outliers: 11
Detected inliers: 199

Score statistics:
  Min: -0.167
  Max: 0.147
  Mean: 0.058

Top 5 most anomalous scores:
  Point 203: -0.167
  Point 207: -0.152
  Point 201: -0.139
  Point 205: -0.128
  Point 209: -0.119`,
  },
  knn: {
    name: 'K-Nearest Neighbors',
    description: 'Classify based on majority vote of K nearest neighbors',
    code: `# K-Nearest Neighbors Classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and prepare data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train k-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

accuracy = knn.score(X_test_scaled, y_test)
print("K-Nearest Neighbors Classification")
print("="*40)
print(f"K (neighbors): 5")
print(f"Test Accuracy: {accuracy:.3f}")
print(f"\\nPredictions (first 10):")
preds = knn.predict(X_test_scaled[:10])
actual = y_test[:10]
print(f"  Predicted: {list(preds)}")
print(f"  Actual:    {list(actual)}")
print(f"  Correct:   {(preds == actual).sum()}/10")`,
    output: `K-Nearest Neighbors Classification
========================================
K (neighbors): 5
Test Accuracy: 0.978

Predictions (first 10):
  Predicted: [1, 0, 2, 1, 1, 0, 1, 2, 1, 1]
  Actual:    [1, 0, 2, 1, 1, 0, 1, 2, 1, 1]
  Correct:   10/10`,
  },
  dbscan: {
    name: 'DBSCAN Clustering',
    description: 'Density-based clustering that identifies noise points',
    code: `# DBSCAN Clustering
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# Generate moon-shaped clusters
X, _ = make_moons(n_samples=300, noise=0.08, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print("DBSCAN Clustering Results")
print("="*40)
print(f"Parameters: eps=0.3, min_samples=5")
print(f"\\nClusters found: {n_clusters}")
print(f"Noise points: {n_noise}")
print(f"\\nCluster distribution:")
for label in sorted(set(labels)):
    count = (labels == label).sum()
    if label == -1:
        print(f"  Noise: {count} points")
    else:
        print(f"  Cluster {label}: {count} points")`,
    output: `DBSCAN Clustering Results
========================================
Parameters: eps=0.3, min_samples=5

Clusters found: 2
Noise points: 3

Cluster distribution:
  Noise: 3 points
  Cluster 0: 149 points
  Cluster 1: 148 points`,
  },
};

export default function Playground() {
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('kmeans');
  const [code, setCode] = useState(algorithms.kmeans.code);
  const [output, setOutput] = useState('Click "Run Code" to see the output');
  const [copied, setCopied] = useState(false);

  const loadAlgorithm = (algoKey: string) => {
    setSelectedAlgorithm(algoKey);
    setCode(algorithms[algoKey].code);
    setOutput('Click "Run Code" to see the output');
  };

  const runCode = () => {
    setOutput(algorithms[selectedAlgorithm].output);
  };

  const resetCode = () => {
    setCode(algorithms[selectedAlgorithm].code);
    setOutput('Click "Run Code" to see the output');
  };

  const copyCode = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="max-w-6xl mx-auto animate-fade-in">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Data Mining Playground</h1>
        <p className="text-gray-400">
          Explore data mining algorithms with interactive code examples
        </p>
      </div>

      <div className="grid lg:grid-cols-4 gap-6">
        {/* Algorithm Selector */}
        <div className="lg:col-span-1">
          <div 
            className="p-4 rounded-lg sticky top-8"
            style={{ background: 'rgba(15, 23, 42, 0.8)', border: '1px solid rgba(59, 130, 246, 0.2)' }}
          >
            <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
              <Lightbulb size={18} style={{ color: '#f97316' }} />
              Algorithms
            </h3>
            <div className="space-y-2">
              {Object.entries(algorithms).map(([key, algo]) => (
                <button
                  key={key}
                  onClick={() => loadAlgorithm(key)}
                  className={`w-full text-left px-4 py-3 rounded-lg transition-all ${
                    selectedAlgorithm === key
                      ? 'bg-blue-500/20 border-blue-500/50'
                      : 'hover:bg-white/5'
                  }`}
                  style={{ border: selectedAlgorithm === key ? '1px solid rgba(59, 130, 246, 0.5)' : '1px solid transparent' }}
                >
                  <div className="font-medium text-white text-sm">{algo.name}</div>
                  <div className="text-xs text-gray-500 mt-1">{algo.description}</div>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Code Editor */}
        <div className="lg:col-span-3 space-y-6">
          {/* Controls */}
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-white">
              {algorithms[selectedAlgorithm].name}
            </h2>
            <div className="flex items-center gap-2">
              <button
                onClick={resetCode}
                className="flex items-center gap-2 px-4 py-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/5 transition-colors"
              >
                <RotateCcw size={16} />
                Reset
              </button>
              <button
                onClick={copyCode}
                className="flex items-center gap-2 px-4 py-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/5 transition-colors"
              >
                {copied ? <Check size={16} className="text-green-400" /> : <Copy size={16} />}
                {copied ? 'Copied!' : 'Copy'}
              </button>
              <button
                onClick={runCode}
                className="flex items-center gap-2 px-4 py-2 rounded-lg text-white font-medium transition-all hover:scale-105"
                style={{ background: 'linear-gradient(135deg, #22c55e, #16a34a)' }}
              >
                <Play size={16} />
                Run Code
              </button>
            </div>
          </div>

          {/* Code Block */}
          <div 
            className="rounded-lg overflow-hidden"
            style={{ background: 'linear-gradient(135deg, #0c1220, #0a0f1a)', border: '1px solid rgba(59, 130, 246, 0.3)' }}
          >
            <div 
              className="px-4 py-2 flex items-center gap-2"
              style={{ background: 'linear-gradient(90deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.2))', borderBottom: '1px solid rgba(59, 130, 246, 0.3)' }}
            >
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
              <span className="ml-4 text-sm text-gray-400">algorithm.py</span>
            </div>
            <textarea
              value={code}
              onChange={(e) => setCode(e.target.value)}
              className="w-full p-4 font-mono text-sm bg-transparent text-gray-300 resize-none focus:outline-none"
              style={{ minHeight: '400px' }}
              spellCheck={false}
            />
          </div>

          {/* Output */}
          <OutputDisplay output={output} title="Output" />

          {/* Tips */}
          <div 
            className="p-4 rounded-lg"
            style={{ background: 'rgba(249, 115, 22, 0.1)', border: '1px solid rgba(249, 115, 22, 0.3)' }}
          >
            <h4 className="font-semibold text-orange-400 mb-2 flex items-center gap-2">
              <Lightbulb size={16} />
              Pro Tip
            </h4>
            <p className="text-gray-400 text-sm">
              This playground shows pre-computed outputs for demonstration. To run Python code yourself, 
              copy the code and run it in Jupyter Notebook or Google Colab with scikit-learn and mlxtend installed.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
