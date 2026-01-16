import { Link } from 'react-router-dom';
import { AlertTriangle, Search, Shield, BarChart3 } from 'lucide-react';
import CodeBlock from '../../components/code/CodeBlock';
import FlowDiagram from '../../components/visualizations/FlowDiagram';

const anomalyTypesDiagram = `
flowchart TB
    A[Anomaly Detection Methods]
    A --> S[Statistical]
    A --> D[Distance-Based]
    A --> Den[Density-Based]
    A --> M[Model-Based]
    
    S --> ZS[Z-Score]
    S --> IQR[IQR Method]
    
    D --> KNN[k-NN Distance]
    
    Den --> LOF[Local Outlier Factor]
    
    M --> IF[Isolation Forest]
    M --> AE[Autoencoders]
    
    style A fill:#1e293b,stroke:#f97316
    style S fill:#1e293b,stroke:#3b82f6
    style D fill:#1e293b,stroke:#8b5cf6
    style Den fill:#1e293b,stroke:#22c55e
    style M fill:#1e293b,stroke:#06b6d4
`;

const statisticalCode = `import numpy as np
from scipy import stats

# Generate data with outliers
np.random.seed(42)
normal_data = np.random.normal(50, 10, 1000)  # Normal distribution
outliers = np.array([150, -30, 200, -50, 180])  # Clear outliers
data = np.concatenate([normal_data, outliers])

print("Statistical Anomaly Detection")
print("="*50)
print(f"Data: {len(data)} points, mean={data.mean():.2f}, std={data.std():.2f}")

# Method 1: Z-Score
print("\\n1. Z-SCORE METHOD")
print("-"*30)
z_scores = np.abs(stats.zscore(data))
threshold = 3  # Points beyond 3 standard deviations
z_outliers = data[z_scores > threshold]
print(f"Threshold: |z| > {threshold}")
print(f"Outliers found: {len(z_outliers)}")
print(f"Outlier values: {z_outliers}")

# Method 2: IQR (Interquartile Range)
print("\\n2. IQR METHOD")
print("-"*30)
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

iqr_outliers = data[(data < lower) | (data > upper)]
print(f"Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
print(f"Valid range: [{lower:.2f}, {upper:.2f}]")
print(f"Outliers found: {len(iqr_outliers)}")

# Method 3: Modified Z-Score (more robust)
print("\\n3. MODIFIED Z-SCORE (MAD-based)")
print("-"*30)
median = np.median(data)
mad = np.median(np.abs(data - median))  # Median Absolute Deviation
modified_z = 0.6745 * (data - median) / mad
mod_threshold = 3.5
mad_outliers = data[np.abs(modified_z) > mod_threshold]
print(f"Median={median:.2f}, MAD={mad:.2f}")
print(f"Outliers found: {len(mad_outliers)}")`;

const lofCode = `from sklearn.neighbors import LocalOutlierFactor
import numpy as np

# Create dataset with clusters and outliers
np.random.seed(42)
# Two clusters
cluster1 = np.random.normal([0, 0], 1, (100, 2))
cluster2 = np.random.normal([5, 5], 1, (100, 2))
# Outliers
outliers = np.array([[10, 10], [-5, -5], [2.5, 10], [10, 2.5]])
X = np.vstack([cluster1, cluster2, outliers])

print("Local Outlier Factor (LOF)")
print("="*50)

# Apply LOF
lof = LocalOutlierFactor(
    n_neighbors=20,      # Number of neighbors
    contamination=0.05,  # Expected proportion of outliers
    novelty=False        # False for outlier detection in training data
)

# Fit and predict (returns -1 for outliers, 1 for inliers)
labels = lof.fit_predict(X)
scores = -lof.negative_outlier_factor_  # Higher = more anomalous

# Results
n_outliers = (labels == -1).sum()
print(f"Parameters: n_neighbors={lof.n_neighbors}, contamination={lof.contamination}")
print(f"\\nOutliers detected: {n_outliers}")
print(f"Inliers: {(labels == 1).sum()}")

# Analyze outlier scores
print(f"\\nLOF Score Statistics:")
print(f"  Min: {scores.min():.3f} (most normal)")
print(f"  Max: {scores.max():.3f} (most anomalous)")
print(f"  Mean: {scores.mean():.3f}")

# Show top anomalies
top_indices = np.argsort(scores)[-5:][::-1]
print(f"\\nTop 5 most anomalous points:")
for idx in top_indices:
    print(f"  Index {idx}: score={scores[idx]:.3f}, point={X[idx]}")

# LOF Interpretation
print("\\n" + "="*50)
print("LOF Score Interpretation:")
print("  LOF ≈ 1: Normal (similar density to neighbors)")
print("  LOF > 1: Potential outlier (lower density)")
print("  LOF >> 1: Strong outlier")`;

const isolationForestCode = `from sklearn.ensemble import IsolationForest
import numpy as np

# Generate data
np.random.seed(42)
X_normal = np.random.randn(1000, 2)  # Normal points
X_outliers = np.random.uniform(low=-4, high=4, size=(50, 2))  # Random outliers
X = np.vstack([X_normal, X_outliers])

print("Isolation Forest")
print("="*50)
print("Concept: Anomalies are 'easier to isolate'")
print("  - Fewer splits needed to isolate outliers")
print("  - Normal points are in dense regions, need more splits")

# Apply Isolation Forest
iso_forest = IsolationForest(
    n_estimators=100,     # Number of trees
    max_samples='auto',   # Samples per tree
    contamination=0.05,   # Expected outlier proportion
    random_state=42
)

# Fit and predict
labels = iso_forest.fit_predict(X)  # -1: outlier, 1: inlier
scores = iso_forest.decision_function(X)  # Lower = more anomalous

# Results
print(f"\\nResults:")
print(f"  Total points: {len(X)}")
print(f"  Detected outliers: {(labels == -1).sum()}")
print(f"  Detected inliers: {(labels == 1).sum()}")

# Score analysis
print(f"\\nAnomaly Scores (decision_function):")
print(f"  Range: [{scores.min():.3f}, {scores.max():.3f}]")
print(f"  Threshold (contamination): ~{np.percentile(scores, 5):.3f}")

# Show top anomalies
top_outliers = np.argsort(scores)[:5]
print(f"\\nTop 5 outliers (lowest scores):")
for idx in top_outliers:
    print(f"  Point {X[idx]}: score={scores[idx]:.3f}")

# Feature importance (for interpretability)
print("\\nNote: Isolation Forest is fast and scalable")
print("  - Works well in high dimensions")
print("  - Doesn't require distance calculations")`;

const falseDiscoveryCode = `import numpy as np
from scipy import stats

# Multiple Hypothesis Testing Problem
print("Avoiding False Discoveries in Data Mining")
print("="*60)

# Simulate testing 1000 hypotheses (e.g., gene expression differences)
np.random.seed(42)
n_tests = 1000
n_true_effects = 50  # 50 features actually have a real effect

# Generate p-values
# Null features: uniform p-values (no real effect)
p_null = np.random.uniform(0, 1, n_tests - n_true_effects)
# Alternative features: smaller p-values (real effect)
p_alt = np.random.beta(1, 10, n_true_effects)  # Skewed toward 0
p_values = np.concatenate([p_null, p_alt])

# Standard approach: alpha = 0.05
alpha = 0.05
significant_naive = (p_values < alpha).sum()
print(f"\\n1. NAIVE APPROACH (alpha = {alpha})")
print(f"   Significant results: {significant_naive}")
print(f"   Expected false positives: {(n_tests - n_true_effects) * alpha:.0f}")

# Bonferroni Correction (very conservative)
alpha_bonferroni = alpha / n_tests
significant_bonf = (p_values < alpha_bonferroni).sum()
print(f"\\n2. BONFERRONI CORRECTION")
print(f"   Adjusted alpha: {alpha_bonferroni:.6f}")
print(f"   Significant results: {significant_bonf}")
print(f"   (Very conservative - may miss true effects)")

# Benjamini-Hochberg (FDR control)
def benjamini_hochberg(p_values, alpha=0.05):
    """Control False Discovery Rate using B-H procedure"""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    
    # B-H threshold
    thresholds = alpha * np.arange(1, n+1) / n
    
    # Find largest p-value that beats its threshold
    below = sorted_p <= thresholds
    if below.any():
        max_idx = np.max(np.where(below)[0])
        p_threshold = sorted_p[max_idx]
    else:
        p_threshold = 0
    
    return p_values <= p_threshold, p_threshold

significant_bh, threshold_bh = benjamini_hochberg(p_values, alpha)
print(f"\\n3. BENJAMINI-HOCHBERG (FDR)")
print(f"   Significant results: {significant_bh.sum()}")
print(f"   FDR threshold: {threshold_bh:.6f}")
print(f"   (Controls proportion of false positives among discoveries)")

# Comparison
print("\\n" + "="*60)
print("SUMMARY: Methods to Control False Discoveries")
print("-"*60)
print(f"{'Method':<25} {'Significant':<15} {'Philosophy'}")
print(f"{'Naive (p < 0.05)':<25} {significant_naive:<15} Family-wise error ignored")
print(f"{'Bonferroni':<25} {significant_bonf:<15} Control ANY false positive")
print(f"{'Benjamini-Hochberg':<25} {significant_bh.sum():<15} Control PROPORTION FP")`;

const statisticalValidationCode = `import numpy as np
from scipy import stats

print("Statistical Validation in Data Mining")
print("="*60)

# Example: Comparing two classification models
np.random.seed(42)
n_samples = 100

# Simulated accuracy scores from cross-validation
model_a_scores = np.random.normal(0.85, 0.03, n_samples)  # Model A
model_b_scores = np.random.normal(0.87, 0.03, n_samples)  # Model B

print("\\nModel Comparison:")
print(f"  Model A: mean={model_a_scores.mean():.4f}, std={model_a_scores.std():.4f}")
print(f"  Model B: mean={model_b_scores.mean():.4f}, std={model_b_scores.std():.4f}")

# 1. Paired t-test (are the models significantly different?)
t_stat, p_value = stats.ttest_rel(model_a_scores, model_b_scores)
print(f"\\n1. PAIRED T-TEST")
print(f"   t-statistic: {t_stat:.4f}")
print(f"   p-value: {p_value:.6f}")
print(f"   Significant at 0.05? {'Yes' if p_value < 0.05 else 'No'}")

# 2. Effect size (Cohen's d)
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx-1)*x.std()**2 + (ny-1)*y.std()**2) / (nx+ny-2))
    return (y.mean() - x.mean()) / pooled_std

d = cohens_d(model_a_scores, model_b_scores)
print(f"\\n2. EFFECT SIZE (Cohen's d)")
print(f"   d = {d:.4f}")
print(f"   Interpretation: ", end="")
if abs(d) < 0.2:
    print("Negligible")
elif abs(d) < 0.5:
    print("Small")
elif abs(d) < 0.8:
    print("Medium")
else:
    print("Large")

# 3. Confidence interval for the difference
diff = model_b_scores - model_a_scores
ci_low, ci_high = stats.t.interval(0.95, len(diff)-1, 
                                    loc=diff.mean(), 
                                    scale=stats.sem(diff))
print(f"\\n3. 95% CONFIDENCE INTERVAL")
print(f"   Mean difference: {diff.mean():.4f}")
print(f"   95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"   Contains 0? {'No (significant)' if ci_low > 0 or ci_high < 0 else 'Yes (not significant)'}")

# 4. Practical significance
print(f"\\n4. PRACTICAL SIGNIFICANCE")
min_meaningful_diff = 0.01  # 1% accuracy difference matters
print(f"   Is |diff| > {min_meaningful_diff}? ", end="")
print(f"{'Yes' if abs(diff.mean()) > min_meaningful_diff else 'No'}")`;

const practicalExampleCode = `# Complete Anomaly Detection Pipeline
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Simulated credit card transaction data
np.random.seed(42)
n_normal = 10000
n_fraud = 100

# Normal transactions
normal = np.column_stack([
    np.random.normal(100, 50, n_normal),   # Transaction amount
    np.random.normal(5, 2, n_normal),      # Transaction frequency
    np.random.normal(50, 20, n_normal),    # Distance from home
])

# Fraudulent transactions (different patterns)
fraud = np.column_stack([
    np.random.normal(500, 200, n_fraud),   # Higher amounts
    np.random.normal(15, 5, n_fraud),      # Higher frequency
    np.random.normal(200, 50, n_fraud),    # Farther from home
])

# Combine data
X = np.vstack([normal, fraud])
y_true = np.array([0]*n_normal + [1]*n_fraud)  # 0=normal, 1=fraud

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Credit Card Fraud Detection Pipeline")
print("="*60)
print(f"Dataset: {n_normal} normal + {n_fraud} fraud transactions")
print(f"Fraud rate: {n_fraud/(n_normal+n_fraud)*100:.2f}%")

# Isolation Forest
iso = IsolationForest(
    contamination=0.01,  # Expected fraud rate
    random_state=42
)
y_pred_iso = iso.fit_predict(X_scaled)
y_pred_iso = (y_pred_iso == -1).astype(int)  # Convert to 0/1

print("\\nIsolation Forest Results:")
print(classification_report(y_true, y_pred_iso, target_names=['Normal', 'Fraud']))

# Analysis
from collections import Counter
print("Confusion Matrix Breakdown:")
tp = ((y_true == 1) & (y_pred_iso == 1)).sum()
fp = ((y_true == 0) & (y_pred_iso == 1)).sum()
fn = ((y_true == 1) & (y_pred_iso == 0)).sum()
tn = ((y_true == 0) & (y_pred_iso == 0)).sum()

print(f"  True Positives (Fraud caught): {tp}")
print(f"  False Positives (Normal flagged): {fp}")
print(f"  False Negatives (Fraud missed): {fn}")
print(f"  True Negatives (Normal cleared): {tn}")

# Cost analysis
cost_per_fraud_missed = 500  # Lost money
cost_per_false_alarm = 10     # Investigation cost
total_cost = fn * cost_per_fraud_missed + fp * cost_per_false_alarm
print(f"\\nCost Analysis:")
print(f"  Cost per missed fraud: ${cost_per_fraud_missed}")
print(f"  Cost per false alarm: ${cost_per_false_alarm}")
print(f"  Total cost: ${total_cost}")`;

export default function Module7_AnomalyDetection() {
  return (
    <div className="max-w-4xl mx-auto animate-fade-in">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-4">
          <div 
            className="p-3 rounded-lg"
            style={{ background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(249, 115, 22, 0.2))' }}
          >
            <AlertTriangle size={28} style={{ color: '#ef4444' }} />
          </div>
          <div>
            <p className="text-sm font-medium" style={{ color: '#ef4444' }}>Module 7</p>
            <h1 className="text-3xl font-bold text-white">Anomaly Detection & Statistical Validation</h1>
          </div>
        </div>
        <p className="text-gray-400 text-lg">
          Identify unusual patterns and validate findings to avoid false discoveries.
        </p>
      </div>

      {/* What is Anomaly Detection */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <Search size={24} style={{ color: '#f97316' }} /> What is Anomaly Detection?
        </h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            <strong className="text-white">Anomaly detection</strong> identifies data points, events, or 
            observations that deviate significantly from the expected pattern. Also called 
            <span className="text-orange-400"> outlier detection</span> or 
            <span className="text-red-400"> novelty detection</span>.
          </p>

          <FlowDiagram chart={anomalyTypesDiagram} title="Anomaly Detection Methods" />

          <div className="grid md:grid-cols-3 gap-4 mt-6">
            <div className="p-4 rounded-lg" style={{ background: 'rgba(59, 130, 246, 0.1)' }}>
              <h4 className="font-semibold text-blue-400 mb-2">🔐 Fraud Detection</h4>
              <p className="text-gray-400 text-sm">Credit card fraud, insurance claims</p>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(139, 92, 246, 0.1)' }}>
              <h4 className="font-semibold text-purple-400 mb-2">🖥️ System Health</h4>
              <p className="text-gray-400 text-sm">Network intrusion, server failures</p>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(34, 197, 94, 0.1)' }}>
              <h4 className="font-semibold text-green-400 mb-2">🏥 Healthcare</h4>
              <p className="text-gray-400 text-sm">Unusual vital signs, disease outbreaks</p>
            </div>
          </div>
        </div>
      </section>

      {/* Statistical Methods */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <BarChart3 size={24} style={{ color: '#3b82f6' }} /> Statistical Methods
        </h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            Statistical methods assume data follows a known distribution and flag points that 
            deviate significantly from it.
          </p>

          <div className="grid md:grid-cols-3 gap-4 mb-6">
            <div className="p-4 rounded-lg" style={{ background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
              <h4 className="font-bold text-blue-400 mb-2">Z-Score</h4>
              <p className="text-gray-400 text-sm">Flag if |z| &gt; 3</p>
              <code className="text-cyan-400 text-xs">z = (x - μ) / σ</code>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(139, 92, 246, 0.1)', border: '1px solid rgba(139, 92, 246, 0.3)' }}>
              <h4 className="font-bold text-purple-400 mb-2">IQR Method</h4>
              <p className="text-gray-400 text-sm">Flag outside [Q1-1.5×IQR, Q3+1.5×IQR]</p>
              <code className="text-cyan-400 text-xs">IQR = Q3 - Q1</code>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(34, 197, 94, 0.1)', border: '1px solid rgba(34, 197, 94, 0.3)' }}>
              <h4 className="font-bold text-green-400 mb-2">Modified Z-Score</h4>
              <p className="text-gray-400 text-sm">Uses Median & MAD (robust)</p>
              <code className="text-cyan-400 text-xs">MAD = median(|x - median|)</code>
            </div>
          </div>

          <CodeBlock code={statisticalCode} language="python" title="statistical_anomaly.py" />
        </div>
      </section>

      {/* Local Outlier Factor */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4">🔵 Local Outlier Factor (LOF)</h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            <strong className="text-white">LOF</strong> detects anomalies based on 
            <span className="text-green-400"> local density</span>. A point is anomalous if its 
            density is significantly lower than its neighbors'.
          </p>

          <div className="p-4 rounded-lg mb-6" style={{ background: 'rgba(34, 197, 94, 0.1)', borderLeft: '4px solid #22c55e' }}>
            <h4 className="font-semibold text-green-400 mb-2">Key Insight</h4>
            <p className="text-gray-400 text-sm">
              LOF handles clusters of different densities. A point might be normal in a sparse cluster 
              but anomalous in a dense cluster at the same global distance.
            </p>
          </div>

          <CodeBlock code={lofCode} language="python" title="local_outlier_factor.py" />
        </div>
      </section>

      {/* Isolation Forest */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4">🌲 Isolation Forest</h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            <strong className="text-white">Isolation Forest</strong> is based on the intuition that 
            anomalies are <span className="text-cyan-400">"easier to isolate"</span>. Random trees 
            require fewer splits to isolate outliers than normal points.
          </p>

          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="p-4 rounded-lg" style={{ background: 'rgba(34, 197, 94, 0.1)' }}>
              <h4 className="font-semibold text-green-400 mb-2">✓ Advantages</h4>
              <ul className="text-gray-400 text-sm space-y-1">
                <li>• Fast: O(n log n)</li>
                <li>• Handles high dimensions well</li>
                <li>• No distance calculations</li>
                <li>• Low memory footprint</li>
              </ul>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(239, 68, 68, 0.1)' }}>
              <h4 className="font-semibold text-red-400 mb-2">✗ Limitations</h4>
              <ul className="text-gray-400 text-sm space-y-1">
                <li>• May struggle with local anomalies</li>
                <li>• Axis-parallel splits only</li>
                <li>• Contamination parameter needed</li>
              </ul>
            </div>
          </div>

          <CodeBlock code={isolationForestCode} language="python" title="isolation_forest.py" />
        </div>
      </section>

      {/* False Discovery */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <Shield size={24} style={{ color: '#eab308' }} /> Avoiding False Discoveries
        </h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            When testing many hypotheses (e.g., thousands of genes, features), some will appear 
            significant by chance. <strong className="text-yellow-400">Multiple testing correction</strong> 
            prevents spurious discoveries.
          </p>

          <div className="p-4 rounded-lg mb-6" style={{ background: 'rgba(234, 179, 8, 0.1)', borderLeft: '4px solid #eab308' }}>
            <h4 className="font-semibold text-yellow-400 mb-2">⚠️ The Multiple Testing Problem</h4>
            <p className="text-gray-400 text-sm">
              At α=0.05, testing 1000 hypotheses expects 50 false positives even with no real effects!
            </p>
          </div>

          <div className="overflow-x-auto mb-6">
            <table>
              <thead>
                <tr>
                  <th>Method</th>
                  <th>Controls</th>
                  <th>Strictness</th>
                  <th>Use When</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="font-medium text-blue-400">Bonferroni</td>
                  <td className="text-gray-400">Family-wise error rate</td>
                  <td className="text-red-400">Very strict</td>
                  <td className="text-gray-400">Few tests, no false positives</td>
                </tr>
                <tr>
                  <td className="font-medium text-purple-400">Benjamini-Hochberg</td>
                  <td className="text-gray-400">False Discovery Rate</td>
                  <td className="text-yellow-400">Moderate</td>
                  <td className="text-gray-400">Many tests, exploratory</td>
                </tr>
                <tr>
                  <td className="font-medium text-green-400">Holm-Bonferroni</td>
                  <td className="text-gray-400">Family-wise error rate</td>
                  <td className="text-orange-400">Less strict than Bonf.</td>
                  <td className="text-gray-400">Step-down procedure</td>
                </tr>
              </tbody>
            </table>
          </div>

          <CodeBlock code={falseDiscoveryCode} language="python" title="false_discovery.py" />
        </div>
      </section>

      {/* Statistical Validation */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4">📊 Statistical Validation</h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            Beyond p-values, proper validation requires considering 
            <span className="text-blue-400"> effect size</span>,
            <span className="text-purple-400"> confidence intervals</span>, and
            <span className="text-green-400"> practical significance</span>.
          </p>

          <CodeBlock code={statisticalValidationCode} language="python" title="statistical_validation.py" />
        </div>
      </section>

      {/* Practical Example */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4">🔐 Practical Example: Fraud Detection</h2>
        <div className="card">
          <CodeBlock code={practicalExampleCode} language="python" title="fraud_detection_pipeline.py" />
        </div>
      </section>

      {/* Key Takeaways */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4">📌 Key Takeaways</h2>
        <div 
          className="p-6 rounded-lg"
          style={{ 
            background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(249, 115, 22, 0.1))',
            border: '1px solid rgba(239, 68, 68, 0.3)'
          }}
        >
          <ul className="space-y-3 text-gray-300">
            <li className="flex items-start gap-3">
              <span className="text-red-400 font-bold">1.</span>
              <span><strong className="text-white">Statistical methods</strong> (Z-score, IQR) work well for univariate, normal data</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-red-400 font-bold">2.</span>
              <span><strong className="text-white">LOF</strong> detects local anomalies using density comparison with neighbors</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-red-400 font-bold">3.</span>
              <span><strong className="text-white">Isolation Forest</strong> is fast and scalable, based on isolation path length</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-red-400 font-bold">4.</span>
              <span><strong className="text-white">Multiple testing</strong> requires correction (Bonferroni, Benjamini-Hochberg)</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-red-400 font-bold">5.</span>
              <span>Report <strong className="text-white">effect size</strong> and <strong className="text-white">confidence intervals</strong>, not just p-values</span>
            </li>
          </ul>
        </div>
      </section>

      {/* Navigation */}
      <div className="flex justify-between items-center pt-8 border-t" style={{ borderColor: 'rgba(59, 130, 246, 0.2)' }}>
        <Link
          to="/module/6"
          className="text-gray-400 hover:text-white flex items-center gap-2"
        >
          <span>←</span>
          Previous: Cluster Analysis
        </Link>
        <Link
          to="/playground"
          className="btn-primary flex items-center gap-2"
        >
          Try the Playground
          <span>→</span>
        </Link>
      </div>
    </div>
  );
}
