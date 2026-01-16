import { useState } from 'react';
import { BookOpen, Search } from 'lucide-react';

const glossaryTerms = [
  // A
  { term: 'Accuracy', definition: 'The proportion of correct predictions (both true positives and true negatives) among the total number of cases examined.', category: 'Evaluation' },
  { term: 'Agglomerative Clustering', definition: 'A bottom-up hierarchical clustering approach where each observation starts as its own cluster, and pairs of clusters are merged as one moves up the hierarchy.', category: 'Clustering' },
  { term: 'Anomaly Detection', definition: 'The identification of rare items, events, or observations which raise suspicions by differing significantly from the majority of the data.', category: 'Anomaly' },
  { term: 'Apriori Algorithm', definition: 'An algorithm for mining frequent itemsets and learning association rules. Uses the principle that all subsets of a frequent itemset must be frequent.', category: 'Association' },
  { term: 'Association Rules', definition: 'Rules that describe relationships between variables in large datasets, typically in the form "if X then Y" with associated metrics like support and confidence.', category: 'Association' },
  { term: 'AUC (Area Under Curve)', definition: 'The area under the ROC curve, representing the probability that a classifier will rank a randomly chosen positive instance higher than a negative one.', category: 'Evaluation' },
  
  // B
  { term: 'Bagging', definition: 'Bootstrap Aggregating - an ensemble method that trains multiple models on random subsets of data (with replacement) and combines their predictions.', category: 'Classification' },
  { term: 'Boosting', definition: 'An ensemble technique that sequentially trains models, with each new model focusing on correcting errors made by previous models.', category: 'Classification' },
  
  // C
  { term: 'Centroid', definition: 'The center point of a cluster, typically calculated as the mean of all points in that cluster.', category: 'Clustering' },
  { term: 'Classification', definition: 'A supervised learning task that predicts the categorical class label of new instances based on a training set of labeled data.', category: 'Classification' },
  { term: 'Clustering', definition: 'An unsupervised learning task that groups similar data points together without predefined labels.', category: 'Clustering' },
  { term: 'Confidence (Association)', definition: 'The probability that a transaction containing item X also contains item Y: P(Y|X) = Support(X∪Y) / Support(X).', category: 'Association' },
  { term: 'Confusion Matrix', definition: 'A table showing the performance of a classification model: True Positives, True Negatives, False Positives, and False Negatives.', category: 'Evaluation' },
  { term: 'Cross-Validation', definition: 'A resampling technique to evaluate models by training on subsets of data and testing on held-out portions, reducing overfitting risk.', category: 'Evaluation' },
  
  // D
  { term: 'Data Mining', definition: 'The process of discovering patterns, correlations, and anomalies within large datasets using machine learning, statistics, and database systems.', category: 'General' },
  { term: 'DBSCAN', definition: 'Density-Based Spatial Clustering of Applications with Noise - clusters points in dense regions and identifies outliers in sparse regions.', category: 'Clustering' },
  { term: 'Decision Tree', definition: 'A flowchart-like model where internal nodes test attributes, branches represent outcomes, and leaf nodes represent class labels.', category: 'Classification' },
  { term: 'Dendrogram', definition: 'A tree-like diagram showing the arrangement of clusters produced by hierarchical clustering.', category: 'Clustering' },
  { term: 'Dimensionality Reduction', definition: 'Techniques to reduce the number of features in a dataset while preserving important information (e.g., PCA).', category: 'Preprocessing' },
  
  // E
  { term: 'Elbow Method', definition: 'A technique to find optimal K in K-Means by plotting inertia vs K and looking for the "elbow" where improvement diminishes.', category: 'Clustering' },
  { term: 'Ensemble Methods', definition: 'Techniques that combine multiple models to produce better predictive performance than any single model.', category: 'Classification' },
  { term: 'Entropy', definition: 'A measure of impurity or uncertainty in a dataset. Maximum entropy (1) means equal class distribution; minimum (0) means pure.', category: 'Classification' },
  
  // F
  { term: 'F1 Score', definition: 'The harmonic mean of precision and recall: 2 × (Precision × Recall) / (Precision + Recall).', category: 'Evaluation' },
  { term: 'False Discovery Rate (FDR)', definition: 'The expected proportion of false positives among all positive results when conducting multiple hypothesis tests.', category: 'Anomaly' },
  { term: 'False Negative (FN)', definition: 'An error where the model incorrectly predicts the negative class for a positive instance (missed detection).', category: 'Evaluation' },
  { term: 'False Positive (FP)', definition: 'An error where the model incorrectly predicts the positive class for a negative instance (false alarm).', category: 'Evaluation' },
  { term: 'Feature Selection', definition: 'The process of selecting a subset of relevant features for model construction.', category: 'Preprocessing' },
  { term: 'FP-Growth', definition: 'Frequent Pattern Growth - an efficient algorithm for mining frequent itemsets without candidate generation using an FP-tree structure.', category: 'Association' },
  
  // G
  { term: 'Gini Impurity', definition: 'A measure of how often a randomly chosen element would be incorrectly classified: Gini = 1 - Σpᵢ².', category: 'Classification' },
  
  // H
  { term: 'Hierarchical Clustering', definition: 'Clustering methods that build a hierarchy of clusters, either bottom-up (agglomerative) or top-down (divisive).', category: 'Clustering' },
  
  // I
  { term: 'Inertia', definition: 'In K-Means, the sum of squared distances of samples to their closest cluster center (within-cluster sum of squares).', category: 'Clustering' },
  { term: 'Information Gain', definition: 'The reduction in entropy achieved by splitting data on a particular attribute. Used to select split attributes in decision trees.', category: 'Classification' },
  { term: 'Isolation Forest', definition: 'An anomaly detection algorithm based on the principle that anomalies are easier to isolate using random partitioning.', category: 'Anomaly' },
  
  // K
  { term: 'K-Means', definition: 'A partitioning clustering algorithm that divides data into K clusters by minimizing within-cluster variance.', category: 'Clustering' },
  { term: 'K-Nearest Neighbors (k-NN)', definition: 'A classification algorithm that assigns labels based on the majority class among the K closest training examples.', category: 'Classification' },
  { term: 'KDD', definition: 'Knowledge Discovery in Databases - the process of discovering useful knowledge from data, including selection, preprocessing, transformation, mining, and evaluation.', category: 'General' },
  
  // L
  { term: 'Lift', definition: 'In association rules, the ratio of confidence to expected confidence: Lift(X→Y) = Conf(X→Y) / Support(Y). Lift > 1 indicates positive association.', category: 'Association' },
  { term: 'Local Outlier Factor (LOF)', definition: 'A density-based anomaly detection method that compares local density of a point to its neighbors.', category: 'Anomaly' },
  
  // M
  { term: 'Min-Max Normalization', definition: 'Scaling features to a fixed range [0,1]: X\' = (X - Xmin) / (Xmax - Xmin).', category: 'Preprocessing' },
  { term: 'Missing Values', definition: 'Absent data in a dataset that must be handled through deletion, imputation, or other techniques.', category: 'Preprocessing' },
  
  // N
  { term: 'Naive Bayes', definition: 'A probabilistic classifier based on Bayes\' theorem with the assumption of feature independence.', category: 'Classification' },
  { term: 'Normalization', definition: 'The process of scaling numeric attributes to a standard range to ensure equal contribution to distance calculations.', category: 'Preprocessing' },
  
  // O
  { term: 'Outlier', definition: 'A data point that differs significantly from other observations, potentially indicating errors or rare events.', category: 'Anomaly' },
  { term: 'Overfitting', definition: 'When a model learns the training data too well, including noise, resulting in poor generalization to new data.', category: 'General' },
  
  // P
  { term: 'PCA', definition: 'Principal Component Analysis - a dimensionality reduction technique that transforms data into uncorrelated principal components.', category: 'Preprocessing' },
  { term: 'Precision', definition: 'The proportion of true positives among all positive predictions: TP / (TP + FP).', category: 'Evaluation' },
  { term: 'Pruning', definition: 'In decision trees, the process of removing branches that provide little predictive power to prevent overfitting.', category: 'Classification' },
  
  // R
  { term: 'Random Forest', definition: 'An ensemble method that builds multiple decision trees on random subsets and averages their predictions.', category: 'Classification' },
  { term: 'Recall', definition: 'The proportion of actual positives correctly identified: TP / (TP + FN). Also called Sensitivity.', category: 'Evaluation' },
  { term: 'ROC Curve', definition: 'Receiver Operating Characteristic curve - a plot of True Positive Rate vs False Positive Rate at various classification thresholds.', category: 'Evaluation' },
  
  // S
  { term: 'Silhouette Score', definition: 'A measure of how similar a point is to its own cluster compared to other clusters. Range: -1 to 1, higher is better.', category: 'Clustering' },
  { term: 'Support (Association)', definition: 'The frequency of an itemset in the dataset: Support(X) = count(X) / N.', category: 'Association' },
  { term: 'Support Vector Machine (SVM)', definition: 'A classification algorithm that finds the optimal hyperplane maximizing the margin between classes.', category: 'Classification' },
  
  // T
  { term: 'Training Set', definition: 'The subset of data used to train a machine learning model.', category: 'General' },
  { term: 'Test Set', definition: 'The subset of data used to evaluate the final model performance.', category: 'General' },
  { term: 'True Positive (TP)', definition: 'A correct prediction where the model correctly identifies a positive instance.', category: 'Evaluation' },
  
  // U
  { term: 'Underfitting', definition: 'When a model is too simple to capture the underlying patterns in the data.', category: 'General' },
  
  // Z
  { term: 'Z-Score', definition: 'A measure of how many standard deviations a value is from the mean: z = (x - μ) / σ.', category: 'Preprocessing' },
];

const categories = ['All', 'General', 'Preprocessing', 'Classification', 'Association', 'Clustering', 'Anomaly', 'Evaluation'];

export default function Glossary() {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All');

  const filteredTerms = glossaryTerms.filter(item => {
    const matchesSearch = item.term.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         item.definition.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'All' || item.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const getCategoryColor = (category: string) => {
    const colors: Record<string, string> = {
      General: '#3b82f6',
      Preprocessing: '#8b5cf6',
      Classification: '#22c55e',
      Association: '#f97316',
      Clustering: '#06b6d4',
      Anomaly: '#ef4444',
      Evaluation: '#eab308',
    };
    return colors[category] || '#64748b';
  };

  return (
    <div className="max-w-4xl mx-auto animate-fade-in">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-4">
          <div 
            className="p-3 rounded-lg"
            style={{ background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.2))' }}
          >
            <BookOpen size={28} style={{ color: '#3b82f6' }} />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-white">Glossary</h1>
            <p className="text-gray-400">Data mining terminology and definitions</p>
          </div>
        </div>
      </div>

      {/* Search and Filter */}
      <div className="mb-8 space-y-4">
        {/* Search */}
        <div 
          className="flex items-center gap-3 px-4 py-3 rounded-lg"
          style={{ background: 'rgba(15, 23, 42, 0.8)', border: '1px solid rgba(59, 130, 246, 0.2)' }}
        >
          <Search size={20} className="text-gray-400" />
          <input
            type="text"
            placeholder="Search terms..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="flex-1 bg-transparent text-white placeholder-gray-500 outline-none"
          />
          {searchTerm && (
            <button 
              onClick={() => setSearchTerm('')}
              className="text-gray-400 hover:text-white"
            >
              ×
            </button>
          )}
        </div>

        {/* Category Filter */}
        <div className="flex flex-wrap gap-2">
          {categories.map(cat => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                selectedCategory === cat
                  ? 'text-white'
                  : 'text-gray-400 hover:text-white hover:bg-white/5'
              }`}
              style={selectedCategory === cat ? {
                background: cat === 'All' 
                  ? 'linear-gradient(135deg, #3b82f6, #8b5cf6)' 
                  : `${getCategoryColor(cat)}30`,
                border: `1px solid ${cat === 'All' ? 'transparent' : getCategoryColor(cat)}`,
              } : {
                border: '1px solid rgba(255, 255, 255, 0.1)',
              }}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      {/* Results Count */}
      <p className="text-gray-400 mb-4">
        Showing {filteredTerms.length} of {glossaryTerms.length} terms
      </p>

      {/* Terms List */}
      <div className="space-y-4">
        {filteredTerms.map((item, index) => (
          <div 
            key={index}
            className="p-5 rounded-lg transition-all hover:border-blue-500/30"
            style={{ 
              background: 'rgba(15, 23, 42, 0.8)', 
              border: '1px solid rgba(59, 130, 246, 0.15)' 
            }}
          >
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-2">
                  <h3 className="text-lg font-semibold text-white">{item.term}</h3>
                  <span 
                    className="px-2 py-0.5 rounded text-xs font-medium"
                    style={{ 
                      background: `${getCategoryColor(item.category)}20`,
                      color: getCategoryColor(item.category)
                    }}
                  >
                    {item.category}
                  </span>
                </div>
                <p className="text-gray-400 leading-relaxed">{item.definition}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Empty State */}
      {filteredTerms.length === 0 && (
        <div className="text-center py-12">
          <p className="text-gray-400">No terms found matching your search.</p>
          <button 
            onClick={() => { setSearchTerm(''); setSelectedCategory('All'); }}
            className="mt-4 text-blue-400 hover:text-blue-300"
          >
            Clear filters
          </button>
        </div>
      )}
    </div>
  );
}
