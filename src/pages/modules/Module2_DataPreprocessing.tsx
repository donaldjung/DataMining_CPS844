import { Link } from 'react-router-dom';
import { Sparkles, Database, Wrench, TrendingUp, Layers } from 'lucide-react';
import CodeBlock from '../../components/code/CodeBlock';
import FlowDiagram from '../../components/visualizations/FlowDiagram';

const dataTypesDiagram = `
flowchart TB
    DT[Data Types]
    DT --> N[Numerical]
    DT --> C[Categorical]
    
    N --> Cont[Continuous]
    N --> Disc[Discrete]
    
    C --> Nom[Nominal]
    C --> Ord[Ordinal]
    C --> Bin[Binary]
    
    style DT fill:#1e293b,stroke:#3b82f6
    style N fill:#1e293b,stroke:#f97316
    style C fill:#1e293b,stroke:#8b5cf6
`;

const preprocessingPipeline = `
flowchart LR
    A[Raw Data] --> B[Data Cleaning]
    B --> C[Data Integration]
    C --> D[Data Transformation]
    D --> E[Data Reduction]
    E --> F[Clean Data]
    
    style A fill:#1e293b,stroke:#ef4444
    style F fill:#1e293b,stroke:#22c55e
`;

const missingValuesCode = `import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Sample dataset with missing values
data = {
    'Age': [25, 30, np.nan, 45, 50, np.nan, 35],
    'Salary': [50000, np.nan, 75000, 80000, np.nan, 60000, 70000],
    'Department': ['IT', 'HR', 'IT', np.nan, 'Finance', 'HR', 'IT']
}
df = pd.DataFrame(data)
print("Original Data:")
print(df)
print(f"\\nMissing values:\\n{df.isnull().sum()}")

# Strategy 1: Remove rows with missing values
df_dropped = df.dropna()
print(f"\\nAfter dropping NaN rows: {len(df_dropped)} rows remain")

# Strategy 2: Fill with mean (numerical columns)
imputer = SimpleImputer(strategy='mean')
df['Age_imputed'] = imputer.fit_transform(df[['Age']])
df['Salary_imputed'] = imputer.fit_transform(df[['Salary']])

# Strategy 3: Fill with mode (categorical columns)
df['Department'].fillna(df['Department'].mode()[0], inplace=True)

print("\\nAfter imputation:")
print(df[['Age', 'Age_imputed', 'Salary', 'Salary_imputed', 'Department']])`;

const normalizationCode = `import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Sample data
data = np.array([[100, 0.001],
                 [200, 0.005],
                 [300, 0.010],
                 [400, 0.020],
                 [500, 0.050]])

print("Original Data:")
print(data)

# Min-Max Normalization (scales to [0, 1])
min_max_scaler = MinMaxScaler()
data_minmax = min_max_scaler.fit_transform(data)
print("\\nMin-Max Normalized (0-1 range):")
print(data_minmax)

# Z-Score Standardization (mean=0, std=1)
standard_scaler = StandardScaler()
data_zscore = standard_scaler.fit_transform(data)
print("\\nZ-Score Standardized (mean=0, std=1):")
print(data_zscore)

# Manual calculation for understanding
print("\\n--- Understanding the formulas ---")
col = data[:, 0]
print(f"Column 1: {col}")
print(f"Min-Max: (x - min) / (max - min)")
print(f"  = ({col[0]} - {col.min()}) / ({col.max()} - {col.min()})")
print(f"  = {(col[0] - col.min()) / (col.max() - col.min()):.3f}")
print(f"Z-Score: (x - mean) / std")
print(f"  = ({col[0]} - {col.mean():.1f}) / {col.std():.1f}")
print(f"  = {(col[0] - col.mean()) / col.std():.3f}")`;

const outlierCode = `import numpy as np
import pandas as pd

# Sample data with outliers
np.random.seed(42)
data = np.concatenate([
    np.random.normal(50, 10, 100),  # Normal data
    np.array([150, 200, -50])        # Outliers
])
df = pd.DataFrame({'value': data})

print(f"Data statistics:")
print(f"  Mean: {df['value'].mean():.2f}")
print(f"  Std: {df['value'].std():.2f}")
print(f"  Min: {df['value'].min():.2f}")
print(f"  Max: {df['value'].max():.2f}")

# Method 1: IQR (Interquartile Range)
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_iqr = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]
print(f"\\nIQR Method:")
print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
print(f"  Outliers found: {len(outliers_iqr)}")

# Method 2: Z-Score
from scipy import stats
z_scores = np.abs(stats.zscore(df['value']))
outliers_zscore = df[z_scores > 3]
print(f"\\nZ-Score Method (threshold=3):")
print(f"  Outliers found: {len(outliers_zscore)}")`;

const pcaCode = `import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load Iris dataset (4 features)
iris = load_iris()
X = iris.data
print(f"Original shape: {X.shape}")
print(f"Original features: {iris.feature_names}")

# Apply PCA to reduce to 2 components
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print(f"\\nReduced shape: {X_reduced.shape}")

# Explained variance
print(f"\\nExplained variance ratio:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var:.3f} ({var*100:.1f}%)")
print(f"  Total: {sum(pca.explained_variance_ratio_):.3f} ({sum(pca.explained_variance_ratio_)*100:.1f}%)")

# Component loadings (how original features contribute)
print(f"\\nPrincipal Component loadings:")
for i, component in enumerate(pca.components_):
    print(f"  PC{i+1}: {[f'{c:.2f}' for c in component]}")`;

export default function Module2_DataPreprocessing() {
  return (
    <div className="max-w-4xl mx-auto animate-fade-in">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-4">
          <div 
            className="p-3 rounded-lg"
            style={{ background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(6, 182, 212, 0.2))' }}
          >
            <Sparkles size={28} style={{ color: '#8b5cf6' }} />
          </div>
          <div>
            <p className="text-sm font-medium" style={{ color: '#8b5cf6' }}>Module 2</p>
            <h1 className="text-3xl font-bold text-white">Data Preprocessing</h1>
          </div>
        </div>
        <p className="text-gray-400 text-lg">
          Learn essential techniques to clean, transform, and prepare raw data for effective mining.
        </p>
      </div>

      {/* Why Preprocessing */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4">🎯 Why Data Preprocessing?</h2>
        <div className="card">
          <div 
            className="p-4 rounded-lg mb-4"
            style={{ background: 'rgba(239, 68, 68, 0.1)', borderLeft: '4px solid #ef4444' }}
          >
            <p className="text-gray-300">
              <strong className="text-red-400">"Garbage In, Garbage Out"</strong> — The quality of data mining 
              results directly depends on the quality of input data. Real-world data is often incomplete, 
              noisy, and inconsistent.
            </p>
          </div>
          
          <FlowDiagram chart={preprocessingPipeline} title="Data Preprocessing Pipeline" />

          <div className="grid md:grid-cols-2 gap-4 mt-6">
            <div className="p-4 rounded-lg" style={{ background: 'rgba(59, 130, 246, 0.1)' }}>
              <h4 className="font-semibold text-blue-400 mb-2">📊 Better Accuracy</h4>
              <p className="text-gray-400 text-sm">Clean data leads to more accurate and reliable models</p>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(139, 92, 246, 0.1)' }}>
              <h4 className="font-semibold text-purple-400 mb-2">⚡ Faster Processing</h4>
              <p className="text-gray-400 text-sm">Reduced data size means faster training and inference</p>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(34, 197, 94, 0.1)' }}>
              <h4 className="font-semibold text-green-400 mb-2">🔍 Better Insights</h4>
              <p className="text-gray-400 text-sm">Transformed data can reveal hidden patterns more clearly</p>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(249, 115, 22, 0.1)' }}>
              <h4 className="font-semibold text-orange-400 mb-2">🛡️ Algorithm Requirements</h4>
              <p className="text-gray-400 text-sm">Many algorithms require specific data formats (e.g., no missing values)</p>
            </div>
          </div>
        </div>
      </section>

      {/* Data Types */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <Database size={24} style={{ color: '#3b82f6' }} /> Data Types
        </h2>
        <div className="card">
          <FlowDiagram chart={dataTypesDiagram} title="Types of Data Attributes" />

          <div className="mt-6 space-y-4">
            <div className="p-4 rounded-lg" style={{ background: 'rgba(249, 115, 22, 0.1)', border: '1px solid rgba(249, 115, 22, 0.3)' }}>
              <h4 className="font-bold text-orange-400 mb-2">Numerical (Quantitative)</h4>
              <ul className="text-gray-300 space-y-2 ml-4">
                <li><strong>Continuous:</strong> Infinite values (e.g., temperature, height, salary)</li>
                <li><strong>Discrete:</strong> Countable values (e.g., age in years, number of children)</li>
              </ul>
            </div>

            <div className="p-4 rounded-lg" style={{ background: 'rgba(139, 92, 246, 0.1)', border: '1px solid rgba(139, 92, 246, 0.3)' }}>
              <h4 className="font-bold text-purple-400 mb-2">Categorical (Qualitative)</h4>
              <ul className="text-gray-300 space-y-2 ml-4">
                <li><strong>Nominal:</strong> No order (e.g., colors, countries, IDs)</li>
                <li><strong>Ordinal:</strong> Ordered categories (e.g., ratings: low/medium/high)</li>
                <li><strong>Binary:</strong> Two values (e.g., yes/no, 0/1, true/false)</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Missing Values */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <Wrench size={24} style={{ color: '#ef4444' }} /> Handling Missing Values
        </h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            Missing values are common in real-world datasets. Choosing the right strategy depends on 
            the nature and amount of missing data.
          </p>

          <div className="overflow-x-auto mb-6">
            <table>
              <thead>
                <tr>
                  <th>Strategy</th>
                  <th>When to Use</th>
                  <th>Pros</th>
                  <th>Cons</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="font-medium text-white">Deletion</td>
                  <td className="text-gray-400">Small % missing, MCAR</td>
                  <td className="text-green-400">Simple, no bias</td>
                  <td className="text-red-400">Loses data</td>
                </tr>
                <tr>
                  <td className="font-medium text-white">Mean/Median</td>
                  <td className="text-gray-400">Numerical, random missing</td>
                  <td className="text-green-400">Preserves mean</td>
                  <td className="text-red-400">Reduces variance</td>
                </tr>
                <tr>
                  <td className="font-medium text-white">Mode</td>
                  <td className="text-gray-400">Categorical data</td>
                  <td className="text-green-400">Simple for categories</td>
                  <td className="text-red-400">Over-represents mode</td>
                </tr>
                <tr>
                  <td className="font-medium text-white">Prediction</td>
                  <td className="text-gray-400">Complex patterns exist</td>
                  <td className="text-green-400">More accurate</td>
                  <td className="text-red-400">Computationally expensive</td>
                </tr>
              </tbody>
            </table>
          </div>

          <CodeBlock code={missingValuesCode} language="python" title="missing_values.py" />
        </div>
      </section>

      {/* Normalization */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <TrendingUp size={24} style={{ color: '#22c55e' }} /> Data Transformation
        </h2>
        <div className="card">
          <h3 className="text-xl font-semibold text-white mb-4">Normalization & Standardization</h3>
          <p className="text-gray-300 mb-4">
            Many algorithms (like k-NN, SVM, neural networks) are sensitive to feature scales. 
            Normalization ensures all features contribute equally.
          </p>

          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="p-4 rounded-lg" style={{ background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
              <h4 className="font-bold text-blue-400 mb-2">Min-Max Normalization</h4>
              <p className="text-gray-400 text-sm mb-2">Scales values to [0, 1] range</p>
              <div className="p-2 rounded font-mono text-sm" style={{ background: 'rgba(0,0,0,0.3)' }}>
                <span className="text-cyan-400">X' = (X - X<sub>min</sub>) / (X<sub>max</sub> - X<sub>min</sub>)</span>
              </div>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(139, 92, 246, 0.1)', border: '1px solid rgba(139, 92, 246, 0.3)' }}>
              <h4 className="font-bold text-purple-400 mb-2">Z-Score Standardization</h4>
              <p className="text-gray-400 text-sm mb-2">Transforms to mean=0, std=1</p>
              <div className="p-2 rounded font-mono text-sm" style={{ background: 'rgba(0,0,0,0.3)' }}>
                <span className="text-cyan-400">X' = (X - μ) / σ</span>
              </div>
            </div>
          </div>

          <CodeBlock code={normalizationCode} language="python" title="normalization.py" />
        </div>
      </section>

      {/* Outliers */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4">🔍 Outlier Detection</h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            Outliers are data points that deviate significantly from other observations. 
            They can be errors or genuinely unusual cases.
          </p>

          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="p-4 rounded-lg" style={{ background: 'rgba(249, 115, 22, 0.1)' }}>
              <h4 className="font-semibold text-orange-400 mb-2">IQR Method</h4>
              <p className="text-gray-400 text-sm">
                Points outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR] are outliers
              </p>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(139, 92, 246, 0.1)' }}>
              <h4 className="font-semibold text-purple-400 mb-2">Z-Score Method</h4>
              <p className="text-gray-400 text-sm">
                Points with |z-score| &gt; 3 are outliers (3σ from mean)
              </p>
            </div>
          </div>

          <CodeBlock code={outlierCode} language="python" title="outlier_detection.py" />
        </div>
      </section>

      {/* Dimensionality Reduction */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <Layers size={24} style={{ color: '#06b6d4' }} /> Dimensionality Reduction
        </h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            High-dimensional data can suffer from the <strong className="text-cyan-400">"curse of dimensionality"</strong>. 
            Reducing features improves performance and helps visualization.
          </p>

          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="p-4 rounded-lg" style={{ background: 'rgba(6, 182, 212, 0.1)', border: '1px solid rgba(6, 182, 212, 0.3)' }}>
              <h4 className="font-bold text-cyan-400 mb-2">Feature Selection</h4>
              <ul className="text-gray-400 text-sm space-y-1">
                <li>• Select subset of original features</li>
                <li>• Filter methods (correlation, chi-square)</li>
                <li>• Wrapper methods (forward/backward selection)</li>
                <li>• Embedded methods (Lasso, tree importance)</li>
              </ul>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(139, 92, 246, 0.1)', border: '1px solid rgba(139, 92, 246, 0.3)' }}>
              <h4 className="font-bold text-purple-400 mb-2">Feature Extraction</h4>
              <ul className="text-gray-400 text-sm space-y-1">
                <li>• Create new features from originals</li>
                <li>• PCA (Principal Component Analysis)</li>
                <li>• LDA (Linear Discriminant Analysis)</li>
                <li>• t-SNE (for visualization)</li>
              </ul>
            </div>
          </div>

          <h3 className="text-xl font-semibold text-white mb-4">Principal Component Analysis (PCA)</h3>
          <CodeBlock code={pcaCode} language="python" title="pca_example.py" />
        </div>
      </section>

      {/* Key Takeaways */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4">📌 Key Takeaways</h2>
        <div 
          className="p-6 rounded-lg"
          style={{ 
            background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(6, 182, 212, 0.1))',
            border: '1px solid rgba(139, 92, 246, 0.3)'
          }}
        >
          <ul className="space-y-3 text-gray-300">
            <li className="flex items-start gap-3">
              <span className="text-purple-400 font-bold">1.</span>
              <span><strong className="text-white">Data quality</strong> directly impacts mining results — "Garbage In, Garbage Out"</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-purple-400 font-bold">2.</span>
              <span><strong className="text-white">Missing values</strong> can be handled by deletion, imputation (mean/median/mode), or prediction</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-purple-400 font-bold">3.</span>
              <span><strong className="text-white">Normalization</strong> (Min-Max) and <strong className="text-white">Standardization</strong> (Z-Score) make features comparable</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-purple-400 font-bold">4.</span>
              <span><strong className="text-white">Outliers</strong> can be detected using IQR or Z-Score methods</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-purple-400 font-bold">5.</span>
              <span><strong className="text-white">Dimensionality reduction</strong> (PCA) combats the curse of dimensionality</span>
            </li>
          </ul>
        </div>
      </section>

      {/* Navigation */}
      <div className="flex justify-between items-center pt-8 border-t" style={{ borderColor: 'rgba(59, 130, 246, 0.2)' }}>
        <Link
          to="/module/1"
          className="text-gray-400 hover:text-white flex items-center gap-2"
        >
          <span>←</span>
          Previous: Introduction
        </Link>
        <Link
          to="/module/3"
          className="btn-primary flex items-center gap-2"
        >
          Next: Classification Basics
          <span>→</span>
        </Link>
      </div>
    </div>
  );
}
