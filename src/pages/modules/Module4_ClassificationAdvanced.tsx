import { Link } from 'react-router-dom';
import { Brain, Target, Layers, TrendingUp, Zap } from 'lucide-react';
import CodeBlock from '../../components/code/CodeBlock';
import FlowDiagram from '../../components/visualizations/FlowDiagram';

const knnDiagram = `
flowchart LR
    A[New Point ?] --> B[Find K Nearest]
    B --> C[Vote by Class]
    C --> D[Assign Label]
    
    style A fill:#1e293b,stroke:#f97316
    style D fill:#1e293b,stroke:#22c55e
`;

const ensembleDiagram = `
flowchart TB
    D[Training Data]
    D --> M1[Model 1]
    D --> M2[Model 2]
    D --> M3[Model 3]
    D --> Mn[Model n]
    
    M1 --> V[Voting/Averaging]
    M2 --> V
    M3 --> V
    Mn --> V
    
    V --> F[Final Prediction]
    
    style D fill:#1e293b,stroke:#3b82f6
    style V fill:#1e293b,stroke:#8b5cf6
    style F fill:#1e293b,stroke:#22c55e
`;

const knnCode = `from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load and split data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# Important: Scale features for distance-based algorithms!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Find optimal K using cross-validation
k_range = range(1, 21)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    k_scores.append(scores.mean())

optimal_k = k_range[np.argmax(k_scores)]
print(f"Optimal K: {optimal_k} (accuracy: {max(k_scores):.3f})")

# Train with optimal K
knn = KNeighborsClassifier(
    n_neighbors=optimal_k,
    weights='uniform',      # or 'distance' for weighted voting
    metric='euclidean'      # or 'manhattan', 'minkowski'
)
knn.fit(X_train_scaled, y_train)

# Evaluate
print(f"\\nTest Accuracy: {knn.score(X_test_scaled, y_test):.3f}")

# Predict with probabilities
sample = X_test_scaled[0:1]
probabilities = knn.predict_proba(sample)
print(f"\\nPrediction probabilities: {probabilities[0]}")`;

const naiveBayesCode = `from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Gaussian Naive Bayes (for continuous features)
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

print("Gaussian Naive Bayes Results:")
print(f"Accuracy: {gnb.score(X_test, y_test):.3f}")
print(f"\\nClass priors (P(class)): {gnb.class_prior_}")

# Example: Text classification with Multinomial NB
from sklearn.feature_extraction.text import CountVectorizer

# Simple text dataset
texts = [
    "buy cheap products sale discount",
    "free money lottery winner claim",
    "meeting tomorrow project deadline",
    "quarterly report financial analysis",
    "limited offer act now special price",
    "team lunch next week planning"
]
labels = [1, 1, 0, 0, 1, 0]  # 1=spam, 0=not spam

# Convert text to features
vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(texts)

mnb = MultinomialNB()
mnb.fit(X_text, labels)

# Predict new text
new_text = ["buy now free discount offer"]
new_X = vectorizer.transform(new_text)
prediction = mnb.predict(new_X)
print(f"\\n'{new_text[0]}' -> {'Spam' if prediction[0] else 'Not Spam'}")`;

const svmCode = `from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np

# Generate sample data
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10,
    n_classes=2, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale features (critical for SVM!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM with different kernels
kernels = ['linear', 'rbf', 'poly']
for kernel in kernels:
    svm = SVC(kernel=kernel, random_state=42)
    svm.fit(X_train_scaled, y_train)
    acc = svm.score(X_test_scaled, y_test)
    print(f"{kernel.upper()} kernel accuracy: {acc:.3f}")

# Hyperparameter tuning with GridSearch
param_grid = {
    'C': [0.1, 1, 10],           # Regularization
    'gamma': ['scale', 'auto'],  # Kernel coefficient
    'kernel': ['rbf']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

print(f"\\nBest parameters: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.3f}")
print(f"Test accuracy: {grid_search.score(X_test_scaled, y_test):.3f}")`;

const ensembleCode = `from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# 1. Random Forest (Bagging with decision trees)
rf = RandomForestClassifier(
    n_estimators=100,     # Number of trees
    max_depth=5,          # Tree depth
    random_state=42
)
rf.fit(X_train, y_train)
print(f"Random Forest Accuracy: {rf.score(X_test, y_test):.3f}")
print(f"Feature Importances: {rf.feature_importances_}")

# 2. Gradient Boosting (Sequential error correction)
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,    # Shrinkage
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)
print(f"\\nGradient Boosting Accuracy: {gb.score(X_test, y_test):.3f}")

# 3. AdaBoost (Adaptive Boosting)
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    random_state=42
)
ada.fit(X_train, y_train)
print(f"AdaBoost Accuracy: {ada.score(X_test, y_test):.3f}")

# 4. Voting Classifier (Combine different models)
voting = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('svc', SVC(probability=True))
    ],
    voting='soft'  # Use probability averaging
)
voting.fit(X_train, y_train)
print(f"\\nVoting Classifier Accuracy: {voting.score(X_test, y_test):.3f}")`;

const crossValCode = `from sklearn.model_selection import (
    cross_val_score, 
    KFold, 
    StratifiedKFold,
    LeaveOneOut
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target

clf = RandomForestClassifier(n_estimators=50, random_state=42)

# 1. Simple K-Fold Cross-Validation
scores = cross_val_score(clf, X, y, cv=5)  # 5-fold CV
print(f"5-Fold CV Scores: {scores}")
print(f"Mean: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

# 2. Stratified K-Fold (preserves class distribution)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stratified_scores = cross_val_score(clf, X, y, cv=skf)
print(f"\\nStratified 5-Fold: {stratified_scores.mean():.3f}")

# 3. Multiple metrics
from sklearn.model_selection import cross_validate
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
results = cross_validate(clf, X, y, cv=5, scoring=scoring)
print(f"\\nMultiple Metrics:")
for metric in scoring:
    key = f'test_{metric}'
    print(f"  {metric}: {results[key].mean():.3f}")`;

const rocCode = `from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Binary classification
X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Get probability predictions
y_proba = clf.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

print(f"ROC AUC Score: {roc_auc:.3f}")
print(f"\\nSample points on ROC curve:")
print(f"{'Threshold':<12} {'FPR':<8} {'TPR':<8}")
for i in range(0, len(thresholds), len(thresholds)//5):
    print(f"{thresholds[i]:<12.3f} {fpr[i]:<8.3f} {tpr[i]:<8.3f}")

# Interpretation
print(f"\\n--- ROC Curve Interpretation ---")
print(f"AUC = 0.5: Random guessing (diagonal line)")
print(f"AUC = 1.0: Perfect classification")
print(f"AUC = {roc_auc:.3f}: Your model performs {'well' if roc_auc > 0.8 else 'moderately'}")`;

export default function Module4_ClassificationAdvanced() {
  return (
    <div className="max-w-4xl mx-auto animate-fade-in">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-4">
          <div 
            className="p-3 rounded-lg"
            style={{ background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(249, 115, 22, 0.2))' }}
          >
            <Brain size={28} style={{ color: '#8b5cf6' }} />
          </div>
          <div>
            <p className="text-sm font-medium" style={{ color: '#8b5cf6' }}>Module 4</p>
            <h1 className="text-3xl font-bold text-white">Classification - Advanced</h1>
          </div>
        </div>
        <p className="text-gray-400 text-lg">
          Explore k-NN, Naive Bayes, SVM, ensemble methods, and advanced evaluation techniques.
        </p>
      </div>

      {/* k-NN */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <Target size={24} style={{ color: '#f97316' }} /> K-Nearest Neighbors (k-NN)
        </h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            <strong className="text-white">k-NN</strong> is a lazy learning algorithm that classifies 
            new instances based on the majority class of their k nearest neighbors in the feature space.
          </p>

          <FlowDiagram chart={knnDiagram} title="k-NN Classification Process" />

          <div className="grid md:grid-cols-2 gap-4 my-6">
            <div className="p-4 rounded-lg" style={{ background: 'rgba(34, 197, 94, 0.1)' }}>
              <h4 className="font-semibold text-green-400 mb-2">✓ Advantages</h4>
              <ul className="text-gray-400 text-sm space-y-1">
                <li>• Simple and intuitive</li>
                <li>• No training phase</li>
                <li>• Naturally handles multi-class</li>
                <li>• Non-parametric (no assumptions)</li>
              </ul>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(239, 68, 68, 0.1)' }}>
              <h4 className="font-semibold text-red-400 mb-2">✗ Disadvantages</h4>
              <ul className="text-gray-400 text-sm space-y-1">
                <li>• Slow prediction (computes all distances)</li>
                <li>• Sensitive to feature scaling</li>
                <li>• Curse of dimensionality</li>
                <li>• Choosing K is critical</li>
              </ul>
            </div>
          </div>

          <CodeBlock code={knnCode} language="python" title="knn_classifier.py" />
        </div>
      </section>

      {/* Naive Bayes */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4">🎲 Naive Bayes Classifier</h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            <strong className="text-white">Naive Bayes</strong> applies Bayes' theorem with the 
            "naive" assumption of feature independence. Despite this simplification, it often 
            performs surprisingly well, especially for text classification.
          </p>

          <div className="p-4 rounded-lg mb-6" style={{ background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
            <h4 className="font-bold text-blue-400 mb-2">Bayes' Theorem</h4>
            <div className="p-3 rounded font-mono text-sm text-center" style={{ background: 'rgba(0,0,0,0.3)' }}>
              <span className="text-cyan-400">P(Class|Features) = P(Features|Class) × P(Class) / P(Features)</span>
            </div>
            <p className="text-gray-400 text-sm mt-2">
              Choose the class with highest posterior probability.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-4 mb-6">
            <div className="p-4 rounded-lg" style={{ background: 'rgba(139, 92, 246, 0.1)' }}>
              <h4 className="font-semibold text-purple-400 mb-2">Gaussian NB</h4>
              <p className="text-gray-400 text-sm">For continuous features (assumes normal distribution)</p>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(249, 115, 22, 0.1)' }}>
              <h4 className="font-semibold text-orange-400 mb-2">Multinomial NB</h4>
              <p className="text-gray-400 text-sm">For discrete counts (text/document classification)</p>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(6, 182, 212, 0.1)' }}>
              <h4 className="font-semibold text-cyan-400 mb-2">Bernoulli NB</h4>
              <p className="text-gray-400 text-sm">For binary features (word presence/absence)</p>
            </div>
          </div>

          <CodeBlock code={naiveBayesCode} language="python" title="naive_bayes.py" />
        </div>
      </section>

      {/* SVM */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <Layers size={24} style={{ color: '#06b6d4' }} /> Support Vector Machines (SVM)
        </h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            <strong className="text-white">SVM</strong> finds the optimal hyperplane that maximizes 
            the margin between classes. It can handle non-linear boundaries using kernel functions.
          </p>

          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="p-4 rounded-lg" style={{ background: 'rgba(6, 182, 212, 0.1)', border: '1px solid rgba(6, 182, 212, 0.3)' }}>
              <h4 className="font-bold text-cyan-400 mb-2">Key Concepts</h4>
              <ul className="text-gray-400 text-sm space-y-1">
                <li><strong className="text-white">Hyperplane:</strong> Decision boundary</li>
                <li><strong className="text-white">Margin:</strong> Distance from boundary to nearest points</li>
                <li><strong className="text-white">Support Vectors:</strong> Points closest to boundary</li>
                <li><strong className="text-white">C parameter:</strong> Trade-off (margin vs errors)</li>
              </ul>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(139, 92, 246, 0.1)', border: '1px solid rgba(139, 92, 246, 0.3)' }}>
              <h4 className="font-bold text-purple-400 mb-2">Kernel Functions</h4>
              <ul className="text-gray-400 text-sm space-y-1">
                <li><strong className="text-white">Linear:</strong> K(x,y) = x·y</li>
                <li><strong className="text-white">Polynomial:</strong> K(x,y) = (x·y + c)^d</li>
                <li><strong className="text-white">RBF (Gaussian):</strong> K(x,y) = exp(-γ||x-y||²)</li>
                <li><strong className="text-white">Sigmoid:</strong> K(x,y) = tanh(αx·y + c)</li>
              </ul>
            </div>
          </div>

          <CodeBlock code={svmCode} language="python" title="svm_classifier.py" />
        </div>
      </section>

      {/* Ensemble Methods */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <Zap size={24} style={{ color: '#eab308' }} /> Ensemble Methods
        </h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            <strong className="text-white">Ensemble methods</strong> combine multiple models to 
            produce better predictions than any single model. The wisdom of crowds!
          </p>

          <FlowDiagram chart={ensembleDiagram} title="Ensemble Learning Architecture" />

          <div className="grid md:grid-cols-2 gap-4 my-6">
            <div className="p-4 rounded-lg" style={{ background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
              <h4 className="font-bold text-blue-400 mb-2">Bagging (Bootstrap Aggregating)</h4>
              <ul className="text-gray-400 text-sm space-y-1">
                <li>• Train models on random subsets (with replacement)</li>
                <li>• Models trained in <strong className="text-white">parallel</strong></li>
                <li>• Reduces <strong className="text-white">variance</strong></li>
                <li>• Example: <strong className="text-white">Random Forest</strong></li>
              </ul>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(249, 115, 22, 0.1)', border: '1px solid rgba(249, 115, 22, 0.3)' }}>
              <h4 className="font-bold text-orange-400 mb-2">Boosting</h4>
              <ul className="text-gray-400 text-sm space-y-1">
                <li>• Models trained <strong className="text-white">sequentially</strong></li>
                <li>• Each focuses on previous errors</li>
                <li>• Reduces <strong className="text-white">bias</strong></li>
                <li>• Examples: <strong className="text-white">AdaBoost, Gradient Boosting, XGBoost</strong></li>
              </ul>
            </div>
          </div>

          <CodeBlock code={ensembleCode} language="python" title="ensemble_methods.py" />
        </div>
      </section>

      {/* Cross-Validation */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4">🔄 Cross-Validation</h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            <strong className="text-white">Cross-validation</strong> provides a more robust estimate 
            of model performance by training and testing on multiple subsets of data.
          </p>

          <div className="p-4 rounded-lg mb-6" style={{ background: 'rgba(34, 197, 94, 0.1)', borderLeft: '4px solid #22c55e' }}>
            <h4 className="font-semibold text-green-400 mb-2">K-Fold Cross-Validation</h4>
            <p className="text-gray-400 text-sm">
              Split data into K folds. For each fold: train on K-1 folds, test on remaining fold.
              Average all K test scores for final estimate.
            </p>
          </div>

          <CodeBlock code={crossValCode} language="python" title="cross_validation.py" />
        </div>
      </section>

      {/* ROC Curves */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <TrendingUp size={24} style={{ color: '#22c55e' }} /> ROC Curves & AUC
        </h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            The <strong className="text-white">ROC (Receiver Operating Characteristic) curve</strong> plots 
            True Positive Rate vs False Positive Rate at various threshold settings. 
            <strong className="text-white"> AUC (Area Under Curve)</strong> summarizes overall performance.
          </p>

          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="p-4 rounded-lg" style={{ background: 'rgba(22, 163, 74, 0.1)' }}>
              <h4 className="font-semibold text-green-500 mb-2">True Positive Rate (TPR)</h4>
              <p className="text-gray-400 text-sm">= Recall = TP / (TP + FN)</p>
              <p className="text-gray-500 text-xs mt-1">How many actual positives caught?</p>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(239, 68, 68, 0.1)' }}>
              <h4 className="font-semibold text-red-400 mb-2">False Positive Rate (FPR)</h4>
              <p className="text-gray-400 text-sm">= FP / (FP + TN)</p>
              <p className="text-gray-500 text-xs mt-1">How many negatives incorrectly flagged?</p>
            </div>
          </div>

          <CodeBlock code={rocCode} language="python" title="roc_auc.py" />
        </div>
      </section>

      {/* Key Takeaways */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4">📌 Key Takeaways</h2>
        <div 
          className="p-6 rounded-lg"
          style={{ 
            background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(249, 115, 22, 0.1))',
            border: '1px solid rgba(139, 92, 246, 0.3)'
          }}
        >
          <ul className="space-y-3 text-gray-300">
            <li className="flex items-start gap-3">
              <span className="text-purple-400 font-bold">1.</span>
              <span><strong className="text-white">k-NN</strong> classifies by majority vote of nearest neighbors (scale features!)</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-purple-400 font-bold">2.</span>
              <span><strong className="text-white">Naive Bayes</strong> uses probability with feature independence assumption</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-purple-400 font-bold">3.</span>
              <span><strong className="text-white">SVM</strong> finds optimal hyperplane; kernels handle non-linear boundaries</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-purple-400 font-bold">4.</span>
              <span><strong className="text-white">Ensembles</strong>: Bagging reduces variance; Boosting reduces bias</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-purple-400 font-bold">5.</span>
              <span>Use <strong className="text-white">cross-validation</strong> and <strong className="text-white">ROC-AUC</strong> for robust evaluation</span>
            </li>
          </ul>
        </div>
      </section>

      {/* Navigation */}
      <div className="flex justify-between items-center pt-8 border-t" style={{ borderColor: 'rgba(59, 130, 246, 0.2)' }}>
        <Link
          to="/module/3"
          className="text-gray-400 hover:text-white flex items-center gap-2"
        >
          <span>←</span>
          Previous: Classification Basics
        </Link>
        <Link
          to="/module/5"
          className="btn-primary flex items-center gap-2"
        >
          Next: Association Analysis
          <span>→</span>
        </Link>
      </div>
    </div>
  );
}
