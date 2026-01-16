import { Link } from 'react-router-dom';
import { Boxes, Target, Layers, TrendingUp } from 'lucide-react';
import CodeBlock from '../../components/code/CodeBlock';
import FlowDiagram from '../../components/visualizations/FlowDiagram';

const clusteringTypesDiagram = `
flowchart TB
    CA[Clustering Algorithms]
    CA --> P[Partitioning]
    CA --> H[Hierarchical]
    CA --> D[Density-Based]
    
    P --> KM[K-Means]
    P --> KMed[K-Medoids]
    
    H --> Agg[Agglomerative]
    H --> Div[Divisive]
    
    D --> DB[DBSCAN]
    D --> OP[OPTICS]
    
    style CA fill:#1e293b,stroke:#3b82f6
    style P fill:#1e293b,stroke:#f97316
    style H fill:#1e293b,stroke:#8b5cf6
    style D fill:#1e293b,stroke:#22c55e
`;

const kmeansDiagram = `
flowchart LR
    A[Initialize K Centroids] --> B[Assign Points to Nearest]
    B --> C[Update Centroids]
    C --> D{Converged?}
    D -->|No| B
    D -->|Yes| E[Final Clusters]
    
    style A fill:#1e293b,stroke:#3b82f6
    style E fill:#1e293b,stroke:#22c55e
`;

const kmeansCode = `from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np

# Generate sample data with 4 clusters
X, y_true = make_blobs(
    n_samples=300, centers=4, 
    cluster_std=0.60, random_state=42
)

print("Data shape:", X.shape)

# K-Means clustering
kmeans = KMeans(
    n_clusters=4,        # Number of clusters
    init='k-means++',    # Smart initialization
    n_init=10,           # Run 10 times with different seeds
    max_iter=300,        # Max iterations per run
    random_state=42
)
kmeans.fit(X)

# Results
print(f"\\nCluster Centers:\\n{kmeans.cluster_centers_}")
print(f"\\nCluster Labels (first 20): {kmeans.labels_[:20]}")
print(f"\\nInertia (SSE): {kmeans.inertia_:.2f}")

# Points per cluster
from collections import Counter
cluster_counts = Counter(kmeans.labels_)
print(f"\\nPoints per cluster:")
for cluster, count in sorted(cluster_counts.items()):
    print(f"  Cluster {cluster}: {count} points")

# Predict new points
new_points = np.array([[0, 0], [4, 4], [-3, -3]])
predictions = kmeans.predict(new_points)
print(f"\\nNew point predictions: {predictions}")`;

const elbowCode = `from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import numpy as np

# Generate data
X, _ = make_blobs(n_samples=500, centers=5, random_state=42)

# Elbow Method: Plot SSE vs K
print("Elbow Method - Finding Optimal K")
print("-" * 40)
print(f"{'K':<5} {'Inertia (SSE)':<15} {'Change':<15}")

inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    
    if k > 2:
        change = inertias[-2] - inertias[-1]
        print(f"{k:<5} {inertias[-1]:<15.2f} {change:<15.2f}")
    else:
        print(f"{k:<5} {inertias[-1]:<15.2f}")

print("\\n→ Look for the 'elbow' where improvement slows dramatically")

# Silhouette Analysis
print("\\n\\nSilhouette Analysis")
print("-" * 40)
print(f"{'K':<5} {'Silhouette Score':<20}")

for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"{k:<5} {score:<20.4f}")

print("\\n→ Higher silhouette score = better cluster separation")
print("   Score range: -1 (worst) to 1 (best)")`;

const hierarchicalCode = `from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs
import numpy as np

# Generate sample data
X, _ = make_blobs(n_samples=50, centers=3, random_state=42)

# Agglomerative Clustering
print("Agglomerative (Bottom-up) Clustering")
print("="*50)

# Different linkage methods
linkages = ['single', 'complete', 'average', 'ward']

for link in linkages:
    clustering = AgglomerativeClustering(
        n_clusters=3,
        linkage=link
    )
    labels = clustering.fit_predict(X)
    
    # Count points per cluster
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    
    print(f"\\n{link.upper()} linkage:")
    print(f"  Cluster sizes: {distribution}")

print("\\n" + "="*50)
print("Linkage Methods Explained:")
print("-"*50)
print("SINGLE:   min distance between clusters (chain effect)")
print("COMPLETE: max distance between clusters (compact)")
print("AVERAGE:  mean distance between all pairs")
print("WARD:     minimizes within-cluster variance (spherical)")

# Dendrogram data (for visualization)
Z = linkage(X, method='ward')
print(f"\\nDendrogram linkage matrix shape: {Z.shape}")
print("Each row: [cluster1, cluster2, distance, n_points]")`;

const dbscanCode = `from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np

# Create dataset with noise and non-spherical clusters
X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
X_blobs, _ = make_blobs(n_samples=100, centers=[[2, 2]], cluster_std=0.3, random_state=42)
X = np.vstack([X_moons, X_blobs])

# Add some noise points
np.random.seed(42)
noise = np.random.uniform(low=-1.5, high=3, size=(20, 2))
X = np.vstack([X, noise])

# Scale features
X_scaled = StandardScaler().fit_transform(X)

# DBSCAN clustering
dbscan = DBSCAN(
    eps=0.3,           # Neighborhood radius
    min_samples=5,     # Min points to form a core point
    metric='euclidean'
)
labels = dbscan.fit_predict(X_scaled)

# Analyze results
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print("DBSCAN Clustering Results")
print("="*50)
print(f"Parameters: eps={dbscan.eps}, min_samples={dbscan.min_samples}")
print(f"\\nClusters found: {n_clusters}")
print(f"Noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")

# Cluster distribution
unique, counts = np.unique(labels, return_counts=True)
print(f"\\nCluster distribution:")
for label, count in zip(unique, counts):
    if label == -1:
        print(f"  Noise: {count} points")
    else:
        print(f"  Cluster {label}: {count} points")

# Core samples
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True
n_core = core_samples_mask.sum()
print(f"\\nCore samples: {n_core}")
print(f"Border samples: {len(labels) - n_core - n_noise}")`;

const evaluationCode = `from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
import numpy as np

# Generate data
X, y_true = make_blobs(n_samples=500, centers=4, random_state=42)

print("Cluster Evaluation Metrics")
print("="*60)

# Compare different algorithms
algorithms = {
    'K-Means': KMeans(n_clusters=4, random_state=42, n_init=10),
    'Hierarchical': AgglomerativeClustering(n_clusters=4),
    'DBSCAN': DBSCAN(eps=1.5, min_samples=5)
}

print(f"\\n{'Algorithm':<15} {'Silhouette':<12} {'Calinski':<12} {'Davies-B':<12}")
print("-"*60)

for name, algo in algorithms.items():
    labels = algo.fit_predict(X)
    
    # Skip if only one cluster or all noise
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        print(f"{name:<15} Insufficient clusters")
        continue
    
    # Filter out noise for metrics
    mask = labels != -1
    if mask.sum() < 2:
        continue
        
    sil = silhouette_score(X[mask], labels[mask])
    ch = calinski_harabasz_score(X[mask], labels[mask])
    db = davies_bouldin_score(X[mask], labels[mask])
    
    print(f"{name:<15} {sil:<12.4f} {ch:<12.2f} {db:<12.4f}")

print("\\n" + "="*60)
print("Metric Interpretation:")
print("-"*60)
print("Silhouette:      [-1, 1] Higher = better separation")
print("Calinski-Harabasz: Higher = denser, well-separated clusters")
print("Davies-Bouldin:  Lower = better (measures cluster similarity)")`;

const realWorldCode = `# Customer Segmentation Example
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Simulated customer data
np.random.seed(42)
n_customers = 500

# Features: [Age, Annual Income ($K), Spending Score (1-100)]
customers = np.vstack([
    # Young, High Income, High Spending
    np.random.normal([25, 80, 80], [5, 10, 10], (100, 3)),
    # Young, Low Income, High Spending
    np.random.normal([25, 35, 75], [5, 8, 8], (100, 3)),
    # Middle Age, Medium Income, Medium Spending
    np.random.normal([45, 55, 50], [10, 15, 15], (150, 3)),
    # Older, High Income, Low Spending
    np.random.normal([55, 90, 25], [8, 12, 10], (80, 3)),
    # Older, Low Income, Low Spending
    np.random.normal([60, 30, 20], [10, 8, 10], (70, 3)),
])

# Clip to valid ranges
customers[:, 0] = np.clip(customers[:, 0], 18, 80)  # Age
customers[:, 1] = np.clip(customers[:, 1], 15, 150)  # Income
customers[:, 2] = np.clip(customers[:, 2], 1, 100)  # Score

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customers)

# Find optimal K
print("Finding optimal number of customer segments...")
from sklearn.metrics import silhouette_score

best_k, best_score = 2, -1
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"  K={k}: Silhouette = {score:.3f}")
    if score > best_score:
        best_k, best_score = k, score

print(f"\\nOptimal K = {best_k}")

# Final clustering
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Analyze segments
print("\\n" + "="*60)
print("CUSTOMER SEGMENTATION ANALYSIS")
print("="*60)

for i in range(best_k):
    segment = customers[labels == i]
    print(f"\\nSegment {i+1} ({len(segment)} customers):")
    print(f"  Avg Age: {segment[:,0].mean():.1f} years")
    print(f"  Avg Income: ${segment[:,1].mean():.0f}K")
    print(f"  Avg Spending Score: {segment[:,2].mean():.1f}")
    
    # Segment description
    age = "Young" if segment[:,0].mean() < 35 else "Middle" if segment[:,0].mean() < 50 else "Older"
    income = "High" if segment[:,1].mean() > 60 else "Medium" if segment[:,1].mean() > 40 else "Low"
    spending = "High" if segment[:,2].mean() > 60 else "Medium" if segment[:,2].mean() > 40 else "Low"
    print(f"  Profile: {age}, {income} Income, {spending} Spender")`;

export default function Module6_ClusterAnalysis() {
  return (
    <div className="max-w-4xl mx-auto animate-fade-in">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-4">
          <div 
            className="p-3 rounded-lg"
            style={{ background: 'linear-gradient(135deg, rgba(6, 182, 212, 0.2), rgba(139, 92, 246, 0.2))' }}
          >
            <Boxes size={28} style={{ color: '#06b6d4' }} />
          </div>
          <div>
            <p className="text-sm font-medium" style={{ color: '#06b6d4' }}>Module 6</p>
            <h1 className="text-3xl font-bold text-white">Cluster Analysis</h1>
          </div>
        </div>
        <p className="text-gray-400 text-lg">
          Learn to discover natural groupings in data using K-Means, Hierarchical Clustering, and DBSCAN.
        </p>
      </div>

      {/* What is Clustering */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <Target size={24} style={{ color: '#f97316' }} /> What is Clustering?
        </h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            <strong className="text-white">Clustering</strong> is an unsupervised learning task that 
            groups similar data points together without predefined labels. The goal is to maximize 
            <span className="text-cyan-400"> intra-cluster similarity</span> while minimizing 
            <span className="text-orange-400"> inter-cluster similarity</span>.
          </p>

          <FlowDiagram chart={clusteringTypesDiagram} title="Types of Clustering Algorithms" />

          <div className="grid md:grid-cols-3 gap-4 mt-6">
            <div className="p-4 rounded-lg" style={{ background: 'rgba(249, 115, 22, 0.1)' }}>
              <h4 className="font-semibold text-orange-400 mb-2">🎯 Partitioning</h4>
              <p className="text-gray-400 text-sm">Divide data into K non-overlapping clusters</p>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(139, 92, 246, 0.1)' }}>
              <h4 className="font-semibold text-purple-400 mb-2">🌳 Hierarchical</h4>
              <p className="text-gray-400 text-sm">Create tree-like cluster hierarchy</p>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(34, 197, 94, 0.1)' }}>
              <h4 className="font-semibold text-green-400 mb-2">🔵 Density-Based</h4>
              <p className="text-gray-400 text-sm">Find dense regions, handles noise</p>
            </div>
          </div>
        </div>
      </section>

      {/* K-Means */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4">🎯 K-Means Clustering</h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            <strong className="text-white">K-Means</strong> partitions data into K clusters by 
            iteratively assigning points to the nearest centroid and updating centroids until convergence.
          </p>

          <FlowDiagram chart={kmeansDiagram} title="K-Means Algorithm" />

          <div className="grid md:grid-cols-2 gap-4 my-6">
            <div className="p-4 rounded-lg" style={{ background: 'rgba(34, 197, 94, 0.1)' }}>
              <h4 className="font-semibold text-green-400 mb-2">✓ Advantages</h4>
              <ul className="text-gray-400 text-sm space-y-1">
                <li>• Simple and fast O(nKt)</li>
                <li>• Scales well to large datasets</li>
                <li>• Guaranteed convergence</li>
              </ul>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(239, 68, 68, 0.1)' }}>
              <h4 className="font-semibold text-red-400 mb-2">✗ Limitations</h4>
              <ul className="text-gray-400 text-sm space-y-1">
                <li>• Must specify K in advance</li>
                <li>• Assumes spherical clusters</li>
                <li>• Sensitive to initialization</li>
                <li>• Affected by outliers</li>
              </ul>
            </div>
          </div>

          <CodeBlock code={kmeansCode} language="python" title="kmeans_clustering.py" />
        </div>
      </section>

      {/* Choosing K */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <TrendingUp size={24} style={{ color: '#eab308' }} /> Choosing Optimal K
        </h2>
        <div className="card">
          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="p-4 rounded-lg" style={{ background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
              <h4 className="font-bold text-blue-400 mb-2">Elbow Method</h4>
              <p className="text-gray-400 text-sm">
                Plot inertia (within-cluster SSE) vs K. The "elbow" point where the curve 
                bends indicates optimal K.
              </p>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(139, 92, 246, 0.1)', border: '1px solid rgba(139, 92, 246, 0.3)' }}>
              <h4 className="font-bold text-purple-400 mb-2">Silhouette Analysis</h4>
              <p className="text-gray-400 text-sm">
                Measures how similar points are to their cluster vs other clusters. 
                Higher score (-1 to 1) = better separation.
              </p>
            </div>
          </div>

          <CodeBlock code={elbowCode} language="python" title="choosing_k.py" />
        </div>
      </section>

      {/* Hierarchical Clustering */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <Layers size={24} style={{ color: '#8b5cf6' }} /> Hierarchical Clustering
        </h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            <strong className="text-white">Hierarchical clustering</strong> creates a tree-like 
            structure (dendrogram) of nested clusters. Can be 
            <span className="text-purple-400"> agglomerative</span> (bottom-up) or 
            <span className="text-cyan-400"> divisive</span> (top-down).
          </p>

          <div className="p-4 rounded-lg mb-6" style={{ background: 'rgba(139, 92, 246, 0.1)', borderLeft: '4px solid #8b5cf6' }}>
            <h4 className="font-semibold text-purple-400 mb-2">Agglomerative (Bottom-Up)</h4>
            <p className="text-gray-400 text-sm">
              1. Start with each point as its own cluster<br />
              2. Repeatedly merge closest pair of clusters<br />
              3. Stop when desired number of clusters reached
            </p>
          </div>

          <div className="overflow-x-auto mb-6">
            <table>
              <thead>
                <tr>
                  <th>Linkage Method</th>
                  <th>Distance Between Clusters</th>
                  <th>Characteristics</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="font-medium text-blue-400">Single</td>
                  <td className="text-gray-400">Minimum distance</td>
                  <td className="text-gray-400">Chains, irregular shapes</td>
                </tr>
                <tr>
                  <td className="font-medium text-purple-400">Complete</td>
                  <td className="text-gray-400">Maximum distance</td>
                  <td className="text-gray-400">Compact, spherical</td>
                </tr>
                <tr>
                  <td className="font-medium text-cyan-400">Average</td>
                  <td className="text-gray-400">Mean of all pairs</td>
                  <td className="text-gray-400">Balanced</td>
                </tr>
                <tr>
                  <td className="font-medium text-green-400">Ward</td>
                  <td className="text-gray-400">Minimize variance increase</td>
                  <td className="text-gray-400">Equal-sized, spherical</td>
                </tr>
              </tbody>
            </table>
          </div>

          <CodeBlock code={hierarchicalCode} language="python" title="hierarchical_clustering.py" />
        </div>
      </section>

      {/* DBSCAN */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4">🔵 DBSCAN</h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            <strong className="text-white">DBSCAN</strong> (Density-Based Spatial Clustering of 
            Applications with Noise) finds clusters of arbitrary shape by identifying dense regions 
            separated by sparse regions.
          </p>

          <div className="grid md:grid-cols-3 gap-4 mb-6">
            <div className="p-4 rounded-lg" style={{ background: 'rgba(6, 182, 212, 0.1)' }}>
              <h4 className="font-semibold text-cyan-400 mb-2">Core Point</h4>
              <p className="text-gray-400 text-sm">Has ≥ min_samples within eps radius</p>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(139, 92, 246, 0.1)' }}>
              <h4 className="font-semibold text-purple-400 mb-2">Border Point</h4>
              <p className="text-gray-400 text-sm">Within eps of a core point</p>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(239, 68, 68, 0.1)' }}>
              <h4 className="font-semibold text-red-400 mb-2">Noise Point</h4>
              <p className="text-gray-400 text-sm">Neither core nor border (outlier)</p>
            </div>
          </div>

          <div className="p-4 rounded-lg mb-6" style={{ background: 'rgba(34, 197, 94, 0.1)', borderLeft: '4px solid #22c55e' }}>
            <h4 className="font-semibold text-green-400 mb-2">Key Parameters</h4>
            <ul className="text-gray-400 text-sm space-y-1">
              <li><strong className="text-white">eps (ε):</strong> Neighborhood radius — smaller = tighter clusters</li>
              <li><strong className="text-white">min_samples:</strong> Minimum points to form core — higher = stricter</li>
            </ul>
          </div>

          <CodeBlock code={dbscanCode} language="python" title="dbscan_clustering.py" />
        </div>
      </section>

      {/* Evaluation */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4">📊 Cluster Evaluation</h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            Since clustering is unsupervised, evaluation relies on 
            <strong className="text-white"> internal metrics</strong> that measure cluster quality 
            without ground truth labels.
          </p>

          <CodeBlock code={evaluationCode} language="python" title="cluster_evaluation.py" />
        </div>
      </section>

      {/* Real-World Application */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4">👥 Customer Segmentation</h2>
        <div className="card">
          <CodeBlock code={realWorldCode} language="python" title="customer_segmentation.py" />
        </div>
      </section>

      {/* Key Takeaways */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4">📌 Key Takeaways</h2>
        <div 
          className="p-6 rounded-lg"
          style={{ 
            background: 'linear-gradient(135deg, rgba(6, 182, 212, 0.1), rgba(139, 92, 246, 0.1))',
            border: '1px solid rgba(6, 182, 212, 0.3)'
          }}
        >
          <ul className="space-y-3 text-gray-300">
            <li className="flex items-start gap-3">
              <span className="text-cyan-400 font-bold">1.</span>
              <span><strong className="text-white">K-Means</strong> partitions into K spherical clusters (fast, needs K)</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-cyan-400 font-bold">2.</span>
              <span>Use <strong className="text-white">Elbow</strong> and <strong className="text-white">Silhouette</strong> methods to choose optimal K</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-cyan-400 font-bold">3.</span>
              <span><strong className="text-white">Hierarchical</strong> creates dendrogram; linkage choice affects shapes</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-cyan-400 font-bold">4.</span>
              <span><strong className="text-white">DBSCAN</strong> finds arbitrary shapes and identifies noise/outliers</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-cyan-400 font-bold">5.</span>
              <span>Evaluate with <strong className="text-white">Silhouette, Calinski-Harabasz, Davies-Bouldin</strong> scores</span>
            </li>
          </ul>
        </div>
      </section>

      {/* Navigation */}
      <div className="flex justify-between items-center pt-8 border-t" style={{ borderColor: 'rgba(59, 130, 246, 0.2)' }}>
        <Link
          to="/module/5"
          className="text-gray-400 hover:text-white flex items-center gap-2"
        >
          <span>←</span>
          Previous: Association Analysis
        </Link>
        <Link
          to="/module/7"
          className="btn-primary flex items-center gap-2"
        >
          Next: Anomaly Detection
          <span>→</span>
        </Link>
      </div>
    </div>
  );
}
