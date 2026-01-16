import { Link } from 'react-router-dom';
import { TreeDeciduous, Target, BarChart3, CheckCircle2 } from 'lucide-react';
import CodeBlock from '../../components/code/CodeBlock';
import FlowDiagram from '../../components/visualizations/FlowDiagram';

const classificationProcessDiagram = `
flowchart LR
    A[Training Data] --> B[Learn Model]
    B --> C[Classification Model]
    C --> D[Apply to New Data]
    D --> E[Predicted Labels]
    
    style A fill:#1e293b,stroke:#3b82f6
    style C fill:#1e293b,stroke:#f97316
    style E fill:#1e293b,stroke:#22c55e
`;

const decisionTreeDiagram = `
flowchart TB
    A["Age < 30?"]
    A -->|Yes| B["Income > 50K?"]
    A -->|No| C["Credit Score?"]
    B -->|Yes| D[Approve ✓]
    B -->|No| E[Deny ✗]
    C -->|Good| F[Approve ✓]
    C -->|Poor| G[Deny ✗]
    
    style A fill:#1e293b,stroke:#3b82f6
    style B fill:#1e293b,stroke:#8b5cf6
    style C fill:#1e293b,stroke:#8b5cf6
    style D fill:#1e293b,stroke:#22c55e
    style E fill:#1e293b,stroke:#ef4444
    style F fill:#1e293b,stroke:#22c55e
    style G fill:#1e293b,stroke:#ef4444
`;

const entropyCode = `import numpy as np
import math

def entropy(labels):
    """Calculate entropy of a label distribution"""
    n = len(labels)
    if n == 0:
        return 0
    
    # Count occurrences of each class
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / n
    
    # Calculate entropy: -sum(p * log2(p))
    ent = -sum(p * math.log2(p) for p in probabilities if p > 0)
    return ent

# Example: Binary classification dataset
labels_high_entropy = ['Yes', 'No', 'Yes', 'No', 'Yes', 'No']  # Mixed
labels_low_entropy = ['Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']  # Mostly one class
labels_pure = ['Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes']        # Pure

print("Entropy Examples:")
print(f"  Mixed labels (3 Yes, 3 No):      {entropy(labels_high_entropy):.3f}")
print(f"  Mostly one (5 Yes, 1 No):        {entropy(labels_low_entropy):.3f}")
print(f"  Pure (6 Yes, 0 No):              {entropy(labels_pure):.3f}")
print(f"  Max entropy (binary): 1.0")

# Information Gain calculation
def information_gain(parent_labels, left_labels, right_labels):
    """Calculate information gain from a split"""
    n = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)
    
    parent_entropy = entropy(parent_labels)
    weighted_child_entropy = (n_left/n * entropy(left_labels) + 
                              n_right/n * entropy(right_labels))
    
    return parent_entropy - weighted_child_entropy

# Example: Split on Age < 30
parent = ['Buy', 'Buy', 'No', 'No', 'Buy', 'No', 'No', 'Buy']
left_split = ['Buy', 'Buy', 'Buy']      # Age < 30
right_split = ['No', 'No', 'No', 'No', 'Buy']  # Age >= 30

ig = information_gain(parent, left_split, right_split)
print(f"\\nInformation Gain from split: {ig:.3f}")`;

const giniCode = `import numpy as np

def gini_impurity(labels):
    """Calculate Gini impurity of a label distribution"""
    n = len(labels)
    if n == 0:
        return 0
    
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / n
    
    # Gini = 1 - sum(p^2)
    gini = 1 - sum(p**2 for p in probabilities)
    return gini

# Compare with Entropy
labels_mixed = ['Yes', 'No', 'Yes', 'No']
labels_skewed = ['Yes', 'Yes', 'Yes', 'No']
labels_pure = ['Yes', 'Yes', 'Yes', 'Yes']

print("Gini Impurity vs Entropy Comparison:")
print(f"{'Distribution':<20} {'Gini':<10} {'Entropy':<10}")
print("-" * 40)
print(f"{'50-50 split':<20} {gini_impurity(labels_mixed):.3f}     1.000")
print(f"{'75-25 split':<20} {gini_impurity(labels_skewed):.3f}     0.811")
print(f"{'Pure':<20} {gini_impurity(labels_pure):.3f}     0.000")`;

const decisionTreeCode = `from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# Create and train decision tree
clf = DecisionTreeClassifier(
    criterion='entropy',  # or 'gini'
    max_depth=3,          # Limit depth to prevent overfitting
    min_samples_split=5,  # Minimum samples to split a node
    min_samples_leaf=2    # Minimum samples in leaf nodes
)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate
print("Decision Tree Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Print tree structure
print("\\nTree Structure:")
tree_rules = export_text(clf, feature_names=iris.feature_names)
print(tree_rules)`;

const evaluationCode = `from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix
)
import numpy as np

# Example predictions
y_true = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
y_pred = [1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1]

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(f"               Predicted")
print(f"               Neg   Pos")
print(f"Actual Neg     {cm[0,0]:<5} {cm[0,1]}")
print(f"       Pos     {cm[1,0]:<5} {cm[1,1]}")

# Extract TP, TN, FP, FN
TN, FP, FN, TP = cm.ravel()
print(f"\\nTP={TP}, TN={TN}, FP={FP}, FN={FN}")

# Calculate metrics
print("\\nEvaluation Metrics:")
print(f"Accuracy  = (TP+TN)/(TP+TN+FP+FN) = {accuracy_score(y_true, y_pred):.3f}")
print(f"Precision = TP/(TP+FP) = {precision_score(y_true, y_pred):.3f}")
print(f"Recall    = TP/(TP+FN) = {recall_score(y_true, y_pred):.3f}")
print(f"F1 Score  = 2*P*R/(P+R) = {f1_score(y_true, y_pred):.3f}")

# When to use which metric?
print("\\n--- Metric Selection Guide ---")
print("• Accuracy: Balanced classes, equal cost of errors")
print("• Precision: When FP is costly (spam detection)")
print("• Recall: When FN is costly (disease detection)")
print("• F1: Balance between Precision and Recall")`;

export default function Module3_ClassificationBasics() {
  return (
    <div className="max-w-4xl mx-auto animate-fade-in">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-4">
          <div 
            className="p-3 rounded-lg"
            style={{ background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(59, 130, 246, 0.2))' }}
          >
            <TreeDeciduous size={28} style={{ color: '#22c55e' }} />
          </div>
          <div>
            <p className="text-sm font-medium" style={{ color: '#22c55e' }}>Module 3</p>
            <h1 className="text-3xl font-bold text-white">Classification - Fundamentals</h1>
          </div>
        </div>
        <p className="text-gray-400 text-lg">
          Master decision trees, entropy, information gain, and model evaluation metrics.
        </p>
      </div>

      {/* What is Classification */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <Target size={24} style={{ color: '#3b82f6' }} /> What is Classification?
        </h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            <strong className="text-white">Classification</strong> is a supervised learning task that 
            predicts the <span className="text-blue-400">categorical class label</span> of new instances 
            based on patterns learned from labeled training data.
          </p>

          <FlowDiagram chart={classificationProcessDiagram} title="Classification Process" />

          <div className="grid md:grid-cols-2 gap-4 mt-6">
            <div className="p-4 rounded-lg" style={{ background: 'rgba(59, 130, 246, 0.1)' }}>
              <h4 className="font-semibold text-blue-400 mb-2">Examples</h4>
              <ul className="text-gray-400 text-sm space-y-1">
                <li>• Email: Spam or Not Spam</li>
                <li>• Loan: Approve or Deny</li>
                <li>• Image: Cat, Dog, or Bird</li>
                <li>• Transaction: Fraud or Legitimate</li>
              </ul>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(139, 92, 246, 0.1)' }}>
              <h4 className="font-semibold text-purple-400 mb-2">Key Terms</h4>
              <ul className="text-gray-400 text-sm space-y-1">
                <li>• <strong>Features (X):</strong> Input attributes</li>
                <li>• <strong>Label (y):</strong> Class to predict</li>
                <li>• <strong>Training:</strong> Learning patterns</li>
                <li>• <strong>Testing:</strong> Evaluating model</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Decision Trees */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <TreeDeciduous size={24} style={{ color: '#22c55e' }} /> Decision Trees
        </h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            A <strong className="text-white">decision tree</strong> is a flowchart-like structure where 
            each internal node tests an attribute, each branch represents an outcome, and each leaf node 
            holds a class label.
          </p>

          <FlowDiagram chart={decisionTreeDiagram} title="Example: Loan Approval Decision Tree" />

          <div className="mt-6 p-4 rounded-lg" style={{ background: 'rgba(34, 197, 94, 0.1)', borderLeft: '4px solid #22c55e' }}>
            <h4 className="font-semibold text-green-400 mb-2">How to Build a Decision Tree?</h4>
            <p className="text-gray-400 text-sm">
              The key question: <em className="text-white">Which attribute should we split on at each node?</em>
              <br />
              Answer: Choose the attribute that best <span className="text-green-400">separates the classes</span> 
              (maximizes information gain or minimizes Gini impurity).
            </p>
          </div>
        </div>
      </section>

      {/* Entropy */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4">📊 Entropy & Information Gain</h2>
        <div className="card">
          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="p-4 rounded-lg" style={{ background: 'rgba(249, 115, 22, 0.1)', border: '1px solid rgba(249, 115, 22, 0.3)' }}>
              <h4 className="font-bold text-orange-400 mb-2">Entropy</h4>
              <p className="text-gray-400 text-sm mb-2">
                Measures the <em>impurity</em> or uncertainty in a dataset
              </p>
              <div className="p-2 rounded font-mono text-sm" style={{ background: 'rgba(0,0,0,0.3)' }}>
                <span className="text-cyan-400">H(S) = -Σ p<sub>i</sub> log<sub>2</sub>(p<sub>i</sub>)</span>
              </div>
              <ul className="text-gray-400 text-xs mt-2">
                <li>• H = 0: Pure (one class)</li>
                <li>• H = 1: Maximum uncertainty (50-50)</li>
              </ul>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
              <h4 className="font-bold text-blue-400 mb-2">Information Gain</h4>
              <p className="text-gray-400 text-sm mb-2">
                Reduction in entropy after splitting on an attribute
              </p>
              <div className="p-2 rounded font-mono text-sm" style={{ background: 'rgba(0,0,0,0.3)' }}>
                <span className="text-cyan-400">IG(S, A) = H(S) - Σ|S<sub>v</sub>|/|S| · H(S<sub>v</sub>)</span>
              </div>
              <ul className="text-gray-400 text-xs mt-2">
                <li>• Higher IG = Better split</li>
                <li>• Used in ID3 and C4.5 algorithms</li>
              </ul>
            </div>
          </div>

          <CodeBlock code={entropyCode} language="python" title="entropy_calculation.py" />
        </div>
      </section>

      {/* Gini Index */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4">📈 Gini Impurity</h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            <strong className="text-white">Gini Impurity</strong> is an alternative to entropy, 
            used in the CART algorithm. It measures the probability of incorrectly classifying 
            a randomly chosen element.
          </p>

          <div className="p-4 rounded-lg mb-6" style={{ background: 'rgba(139, 92, 246, 0.1)', border: '1px solid rgba(139, 92, 246, 0.3)' }}>
            <h4 className="font-bold text-purple-400 mb-2">Gini Formula</h4>
            <div className="p-2 rounded font-mono text-sm" style={{ background: 'rgba(0,0,0,0.3)' }}>
              <span className="text-cyan-400">Gini(S) = 1 - Σ p<sub>i</sub><sup>2</sup></span>
            </div>
            <ul className="text-gray-400 text-xs mt-2">
              <li>• Gini = 0: Pure node (one class)</li>
              <li>• Gini = 0.5: Maximum impurity for binary (50-50)</li>
            </ul>
          </div>

          <CodeBlock code={giniCode} language="python" title="gini_impurity.py" />

          <div className="mt-6 p-4 rounded-lg" style={{ background: 'rgba(59, 130, 246, 0.1)' }}>
            <h4 className="font-semibold text-blue-400 mb-2">Entropy vs Gini</h4>
            <p className="text-gray-400 text-sm">
              Both measure impurity and usually produce similar trees. Gini is computationally 
              faster (no logarithm), while entropy is more theoretically grounded. CART uses Gini; 
              ID3/C4.5 use entropy.
            </p>
          </div>
        </div>
      </section>

      {/* Building a Decision Tree */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4">🛠️ Building Decision Trees with sklearn</h2>
        <div className="card">
          <CodeBlock code={decisionTreeCode} language="python" title="decision_tree_sklearn.py" />

          <div className="mt-6 p-4 rounded-lg" style={{ background: 'rgba(234, 179, 8, 0.1)', borderLeft: '4px solid #eab308' }}>
            <h4 className="font-semibold text-yellow-400 mb-2">⚠️ Overfitting in Decision Trees</h4>
            <p className="text-gray-400 text-sm">
              Without constraints, trees can grow too deep and <em>memorize</em> the training data 
              instead of learning general patterns. Solutions:
            </p>
            <ul className="text-gray-400 text-sm mt-2 space-y-1">
              <li>• <strong className="text-white">max_depth:</strong> Limit tree depth</li>
              <li>• <strong className="text-white">min_samples_split:</strong> Minimum samples to split</li>
              <li>• <strong className="text-white">min_samples_leaf:</strong> Minimum samples per leaf</li>
              <li>• <strong className="text-white">Pruning:</strong> Remove branches that don't improve validation accuracy</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Model Evaluation */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <BarChart3 size={24} style={{ color: '#f97316' }} /> Model Evaluation
        </h2>
        <div className="card">
          <h3 className="text-xl font-semibold text-white mb-4">Confusion Matrix</h3>
          
          <div className="overflow-x-auto mb-6">
            <table className="w-full max-w-md mx-auto">
              <thead>
                <tr>
                  <th></th>
                  <th className="text-center">Predicted Negative</th>
                  <th className="text-center">Predicted Positive</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="font-medium">Actual Negative</td>
                  <td className="text-center p-4 rounded" style={{ background: 'rgba(34, 197, 94, 0.2)' }}>
                    <strong className="text-green-400">TN</strong><br />
                    <span className="text-xs text-gray-400">True Negative</span>
                  </td>
                  <td className="text-center p-4 rounded" style={{ background: 'rgba(239, 68, 68, 0.2)' }}>
                    <strong className="text-red-400">FP</strong><br />
                    <span className="text-xs text-gray-400">False Positive</span>
                  </td>
                </tr>
                <tr>
                  <td className="font-medium">Actual Positive</td>
                  <td className="text-center p-4 rounded" style={{ background: 'rgba(239, 68, 68, 0.2)' }}>
                    <strong className="text-red-400">FN</strong><br />
                    <span className="text-xs text-gray-400">False Negative</span>
                  </td>
                  <td className="text-center p-4 rounded" style={{ background: 'rgba(34, 197, 94, 0.2)' }}>
                    <strong className="text-green-400">TP</strong><br />
                    <span className="text-xs text-gray-400">True Positive</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="p-4 rounded-lg" style={{ background: 'rgba(59, 130, 246, 0.1)' }}>
              <h4 className="font-semibold text-blue-400 mb-2">Accuracy</h4>
              <p className="text-gray-400 text-sm">Overall correct predictions</p>
              <code className="text-cyan-400 text-sm">(TP + TN) / Total</code>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(139, 92, 246, 0.1)' }}>
              <h4 className="font-semibold text-purple-400 mb-2">Precision</h4>
              <p className="text-gray-400 text-sm">Of predicted positives, how many correct?</p>
              <code className="text-cyan-400 text-sm">TP / (TP + FP)</code>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(249, 115, 22, 0.1)' }}>
              <h4 className="font-semibold text-orange-400 mb-2">Recall (Sensitivity)</h4>
              <p className="text-gray-400 text-sm">Of actual positives, how many caught?</p>
              <code className="text-cyan-400 text-sm">TP / (TP + FN)</code>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(34, 197, 94, 0.1)' }}>
              <h4 className="font-semibold text-green-400 mb-2">F1 Score</h4>
              <p className="text-gray-400 text-sm">Harmonic mean of Precision & Recall</p>
              <code className="text-cyan-400 text-sm">2 * P * R / (P + R)</code>
            </div>
          </div>

          <CodeBlock code={evaluationCode} language="python" title="evaluation_metrics.py" />
        </div>
      </section>

      {/* Key Takeaways */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4">📌 Key Takeaways</h2>
        <div 
          className="p-6 rounded-lg"
          style={{ 
            background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(59, 130, 246, 0.1))',
            border: '1px solid rgba(34, 197, 94, 0.3)'
          }}
        >
          <ul className="space-y-3 text-gray-300">
            <li className="flex items-start gap-3">
              <CheckCircle2 size={18} className="text-green-400 mt-1" />
              <span><strong className="text-white">Classification</strong> predicts categorical labels from labeled training data</span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 size={18} className="text-green-400 mt-1" />
              <span><strong className="text-white">Decision Trees</strong> use if-then rules learned from attribute splits</span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 size={18} className="text-green-400 mt-1" />
              <span><strong className="text-white">Entropy</strong> measures impurity; <strong className="text-white">Information Gain</strong> guides splits</span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 size={18} className="text-green-400 mt-1" />
              <span><strong className="text-white">Gini Impurity</strong> is an alternative measure used in CART</span>
            </li>
            <li className="flex items-start gap-3">
              <CheckCircle2 size={18} className="text-green-400 mt-1" />
              <span>Evaluate with <strong className="text-white">Accuracy, Precision, Recall, F1</strong> based on business needs</span>
            </li>
          </ul>
        </div>
      </section>

      {/* Navigation */}
      <div className="flex justify-between items-center pt-8 border-t" style={{ borderColor: 'rgba(59, 130, 246, 0.2)' }}>
        <Link
          to="/module/2"
          className="text-gray-400 hover:text-white flex items-center gap-2"
        >
          <span>←</span>
          Previous: Data Preprocessing
        </Link>
        <Link
          to="/module/4"
          className="btn-primary flex items-center gap-2"
        >
          Next: Classification Advanced
          <span>→</span>
        </Link>
      </div>
    </div>
  );
}
