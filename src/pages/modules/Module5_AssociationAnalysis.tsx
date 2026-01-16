import { Link } from 'react-router-dom';
import { Link as LinkIcon, ShoppingCart, TrendingUp, Database } from 'lucide-react';
import CodeBlock from '../../components/code/CodeBlock';
import FlowDiagram from '../../components/visualizations/FlowDiagram';

const aprioriDiagram = `
flowchart TB
    A[Transactions] --> B[Find Frequent 1-itemsets]
    B --> C[Generate Candidate 2-itemsets]
    C --> D[Prune by Min Support]
    D --> E[Find Frequent 2-itemsets]
    E --> F[Generate Candidate 3-itemsets]
    F --> G[...]
    G --> H[No more candidates]
    H --> I[Generate Association Rules]
    
    style A fill:#1e293b,stroke:#3b82f6
    style I fill:#1e293b,stroke:#22c55e
`;

const fpGrowthDiagram = `
flowchart LR
    A[Transactions] --> B[Build FP-Tree]
    B --> C[Mine Patterns]
    C --> D[Association Rules]
    
    style A fill:#1e293b,stroke:#3b82f6
    style B fill:#1e293b,stroke:#8b5cf6
    style D fill:#1e293b,stroke:#22c55e
`;

const basicMetricsCode = `# Association Rule Metrics Explained

# Sample transaction database
transactions = [
    ['bread', 'milk'],
    ['bread', 'diapers', 'beer', 'eggs'],
    ['milk', 'diapers', 'beer', 'cola'],
    ['bread', 'milk', 'diapers', 'beer'],
    ['bread', 'milk', 'diapers', 'cola']
]

total_transactions = len(transactions)  # 5

# Count item occurrences
from collections import Counter
item_counts = Counter()
for trans in transactions:
    for item in trans:
        item_counts[item] += 1

print("Item Frequencies:")
for item, count in item_counts.most_common():
    print(f"  {item}: {count}/5 = {count/total_transactions:.2f}")

# Calculate Support for itemsets
def support(itemset, transactions):
    """Support = (transactions containing itemset) / total transactions"""
    count = sum(1 for t in transactions if set(itemset).issubset(set(t)))
    return count / len(transactions)

# Examples
print(f"\\nSupport Examples:")
print(f"  Support(bread) = {support(['bread'], transactions):.2f}")
print(f"  Support(milk) = {support(['milk'], transactions):.2f}")
print(f"  Support(bread, milk) = {support(['bread', 'milk'], transactions):.2f}")
print(f"  Support(beer, diapers) = {support(['beer', 'diapers'], transactions):.2f}")

# Confidence: P(Y|X) = Support(X,Y) / Support(X)
def confidence(X, Y, transactions):
    """Confidence = Support(X ∪ Y) / Support(X)"""
    return support(X + Y, transactions) / support(X, transactions)

print(f"\\nConfidence Examples:")
print(f"  Confidence(bread → milk) = {confidence(['bread'], ['milk'], transactions):.2f}")
print(f"  Confidence(milk → bread) = {confidence(['milk'], ['bread'], transactions):.2f}")
print(f"  Confidence(diapers → beer) = {confidence(['diapers'], ['beer'], transactions):.2f}")

# Lift: Does X increase likelihood of Y?
def lift(X, Y, transactions):
    """Lift = Confidence(X→Y) / Support(Y) = Support(X,Y) / (Support(X) * Support(Y))"""
    return confidence(X, Y, transactions) / support(Y, transactions)

print(f"\\nLift Examples:")
print(f"  Lift(bread → milk) = {lift(['bread'], ['milk'], transactions):.3f}")
print(f"  Lift(diapers → beer) = {lift(['diapers'], ['beer'], transactions):.3f}")
print("\\n  Lift > 1: Positive association")
print("  Lift = 1: Independent")
print("  Lift < 1: Negative association")`;

const aprioriCode = `# Apriori Algorithm Implementation with mlxtend
# Install: pip install mlxtend

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

# Sample transaction data
transactions = [
    ['bread', 'milk'],
    ['bread', 'diapers', 'beer', 'eggs'],
    ['milk', 'diapers', 'beer', 'cola'],
    ['bread', 'milk', 'diapers', 'beer'],
    ['bread', 'milk', 'diapers', 'cola'],
    ['bread', 'milk', 'beer'],
    ['bread', 'diapers', 'beer'],
    ['milk', 'diapers', 'cola'],
    ['bread', 'milk', 'diapers', 'beer', 'cola'],
    ['bread', 'eggs']
]

# Convert to one-hot encoded format
te = TransactionEncoder()
te_array = te.fit_transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

print("Transaction Matrix (one-hot encoded):")
print(df.head())

# Step 1: Find frequent itemsets (min_support = 0.3 = 30%)
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
print(f"\\nFrequent Itemsets (support >= 30%):")
print(frequent_itemsets.sort_values('support', ascending=False))

# Step 2: Generate association rules
rules = association_rules(
    frequent_itemsets, 
    metric="confidence", 
    min_threshold=0.6  # min confidence = 60%
)

print(f"\\nAssociation Rules (confidence >= 60%):")
rules_display = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
print(rules_display.sort_values('lift', ascending=False).head(10))

# Filter rules by multiple criteria
strong_rules = rules[
    (rules['confidence'] >= 0.7) & 
    (rules['lift'] >= 1.0) &
    (rules['support'] >= 0.2)
]
print(f"\\nStrong Rules (conf>=70%, lift>=1, support>=20%):")
print(strong_rules[['antecedents', 'consequents', 'confidence', 'lift']])`;

const fpGrowthCode = `# FP-Growth: Faster than Apriori (no candidate generation)
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import time

# Larger dataset for comparison
import random
random.seed(42)

items = ['bread', 'milk', 'eggs', 'butter', 'cheese', 'beer', 
         'diapers', 'cola', 'chips', 'yogurt']

# Generate 1000 transactions
transactions = []
for _ in range(1000):
    n_items = random.randint(2, 6)
    trans = random.sample(items, n_items)
    transactions.append(trans)

# Convert to DataFrame
te = TransactionEncoder()
df = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)

# Compare Apriori vs FP-Growth
from mlxtend.frequent_patterns import apriori

print("Performance Comparison (1000 transactions):")

# Apriori timing
start = time.time()
freq_apriori = apriori(df, min_support=0.1, use_colnames=True)
apriori_time = time.time() - start
print(f"Apriori: {apriori_time:.4f}s, {len(freq_apriori)} itemsets")

# FP-Growth timing
start = time.time()
freq_fpgrowth = fpgrowth(df, min_support=0.1, use_colnames=True)
fpgrowth_time = time.time() - start
print(f"FP-Growth: {fpgrowth_time:.4f}s, {len(freq_fpgrowth)} itemsets")

print(f"\\nSpeedup: {apriori_time/fpgrowth_time:.2f}x faster")

# Generate rules from FP-Growth results
rules = association_rules(freq_fpgrowth, metric="lift", min_threshold=1.0)
print(f"\\nTop 5 Rules by Lift:")
top_rules = rules.nlargest(5, 'lift')[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
print(top_rules)`;

const sequentialPatternCode = `# Sequential Pattern Mining Example
# Finding patterns in ordered sequences (e.g., customer purchase over time)

# Simplified sequential pattern example
purchase_sequences = [
    ['TV', 'DVD Player', 'Sound System'],
    ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
    ['TV', 'Sound System', 'Gaming Console'],
    ['Laptop', 'Mouse', 'External Drive'],
    ['TV', 'DVD Player', 'Gaming Console'],
]

# Count 2-item sequential patterns
from collections import defaultdict

def find_sequential_patterns(sequences, min_support=0.4):
    """Find frequent 2-item sequential patterns"""
    n = len(sequences)
    pattern_counts = defaultdict(int)
    
    for seq in sequences:
        # Find all consecutive pairs
        for i in range(len(seq) - 1):
            pattern = (seq[i], seq[i+1])
            pattern_counts[pattern] += 1
    
    # Filter by minimum support
    frequent_patterns = {
        pattern: count/n 
        for pattern, count in pattern_counts.items() 
        if count/n >= min_support
    }
    
    return frequent_patterns

patterns = find_sequential_patterns(purchase_sequences, min_support=0.3)

print("Sequential Patterns (support >= 30%):")
print("Pattern: A → B means 'A is often followed by B'")
print("-" * 40)
for (item1, item2), support in sorted(patterns.items(), key=lambda x: -x[1]):
    print(f"  {item1} → {item2}: support = {support:.2f}")

# Real-world interpretation
print("\\nBusiness Insight:")
print("Customers who buy a TV often buy Sound System next")
print("→ Recommend Sound System after TV purchase")`;

const realWorldCode = `# Real-World Application: Market Basket Analysis

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Simulated grocery store transactions
grocery_transactions = [
    ['milk', 'bread', 'butter'],
    ['milk', 'bread', 'eggs', 'butter'],
    ['beer', 'chips', 'diapers'],
    ['milk', 'bread', 'cereal'],
    ['beer', 'chips', 'pizza'],
    ['milk', 'bread', 'butter', 'cheese'],
    ['beer', 'chips', 'diapers', 'pizza'],
    ['milk', 'cereal', 'juice'],
    ['bread', 'butter', 'jam'],
    ['beer', 'chips'],
    ['milk', 'bread', 'eggs'],
    ['beer', 'pizza', 'chips'],
]

# Process data
te = TransactionEncoder()
df = pd.DataFrame(te.fit_transform(grocery_transactions), columns=te.columns_)

# Mine frequent patterns
frequent = fpgrowth(df, min_support=0.25, use_colnames=True)
rules = association_rules(frequent, metric="lift", min_threshold=1.0)

# Analyze results
print("="*60)
print("MARKET BASKET ANALYSIS REPORT")
print("="*60)

print("\\n📊 Most Frequent Item Combinations:")
for _, row in frequent.nlargest(5, 'support').iterrows():
    items = ', '.join(list(row['itemsets']))
    print(f"  • {items}: {row['support']*100:.1f}% of transactions")

print("\\n🔗 Strongest Associations (Lift > 1):")
for _, row in rules.nlargest(5, 'lift').iterrows():
    ant = ', '.join(list(row['antecedents']))
    con = ', '.join(list(row['consequents']))
    print(f"  • {ant} → {con}")
    print(f"    Confidence: {row['confidence']*100:.1f}%, Lift: {row['lift']:.2f}")

print("\\n💡 Business Recommendations:")
print("  1. Place beer, chips, and diapers near each other")
print("  2. Create 'breakfast bundle': milk + bread + butter")
print("  3. Cross-promote pizza with beer and chips purchases")`;

export default function Module5_AssociationAnalysis() {
  return (
    <div className="max-w-4xl mx-auto animate-fade-in">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-4">
          <div 
            className="p-3 rounded-lg"
            style={{ background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(34, 197, 94, 0.2))' }}
          >
            <LinkIcon size={28} style={{ color: '#3b82f6' }} />
          </div>
          <div>
            <p className="text-sm font-medium" style={{ color: '#3b82f6' }}>Module 5</p>
            <h1 className="text-3xl font-bold text-white">Association Analysis</h1>
          </div>
        </div>
        <p className="text-gray-400 text-lg">
          Discover relationships between items using market basket analysis, Apriori, and FP-Growth algorithms.
        </p>
      </div>

      {/* Market Basket Analysis */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <ShoppingCart size={24} style={{ color: '#f97316' }} /> Market Basket Analysis
        </h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            <strong className="text-white">Association analysis</strong> finds interesting relationships 
            (associations) between items in large datasets. The classic application is 
            <span className="text-orange-400"> market basket analysis</span>: "Customers who buy X also buy Y."
          </p>

          <div className="p-4 rounded-lg mb-4" style={{ background: 'rgba(249, 115, 22, 0.1)', borderLeft: '4px solid #f97316' }}>
            <p className="text-gray-300">
              <strong className="text-orange-400">Famous Example:</strong> "Beer and Diapers" — Analysis 
              revealed that men buying diapers often bought beer, leading to strategic store placement.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="p-4 rounded-lg" style={{ background: 'rgba(59, 130, 246, 0.1)' }}>
              <h4 className="font-semibold text-blue-400 mb-2">Retail</h4>
              <p className="text-gray-400 text-sm">Product placement, cross-selling, promotions</p>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(139, 92, 246, 0.1)' }}>
              <h4 className="font-semibold text-purple-400 mb-2">Web Mining</h4>
              <p className="text-gray-400 text-sm">Page recommendations, click patterns</p>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(34, 197, 94, 0.1)' }}>
              <h4 className="font-semibold text-green-400 mb-2">Healthcare</h4>
              <p className="text-gray-400 text-sm">Symptom-disease relationships, drug interactions</p>
            </div>
          </div>
        </div>
      </section>

      {/* Key Metrics */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <TrendingUp size={24} style={{ color: '#22c55e' }} /> Support, Confidence & Lift
        </h2>
        <div className="card">
          <div className="grid md:grid-cols-3 gap-4 mb-6">
            <div className="p-4 rounded-lg" style={{ background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
              <h4 className="font-bold text-blue-400 mb-2">Support</h4>
              <p className="text-gray-400 text-sm mb-2">How often does the itemset appear?</p>
              <div className="p-2 rounded font-mono text-xs" style={{ background: 'rgba(0,0,0,0.3)' }}>
                <span className="text-cyan-400">Support(X) = P(X) = count(X) / N</span>
              </div>
              <p className="text-gray-500 text-xs mt-2">Filters rare combinations</p>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(139, 92, 246, 0.1)', border: '1px solid rgba(139, 92, 246, 0.3)' }}>
              <h4 className="font-bold text-purple-400 mb-2">Confidence</h4>
              <p className="text-gray-400 text-sm mb-2">Given X, how likely is Y?</p>
              <div className="p-2 rounded font-mono text-xs" style={{ background: 'rgba(0,0,0,0.3)' }}>
                <span className="text-cyan-400">Conf(X→Y) = P(Y|X) = Sup(X,Y)/Sup(X)</span>
              </div>
              <p className="text-gray-500 text-xs mt-2">Measures rule strength</p>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(34, 197, 94, 0.1)', border: '1px solid rgba(34, 197, 94, 0.3)' }}>
              <h4 className="font-bold text-green-400 mb-2">Lift</h4>
              <p className="text-gray-400 text-sm mb-2">How much does X increase Y's likelihood?</p>
              <div className="p-2 rounded font-mono text-xs" style={{ background: 'rgba(0,0,0,0.3)' }}>
                <span className="text-cyan-400">Lift(X→Y) = Conf(X→Y) / Sup(Y)</span>
              </div>
              <p className="text-gray-500 text-xs mt-2">&gt;1: positive, =1: independent, &lt;1: negative</p>
            </div>
          </div>

          <CodeBlock code={basicMetricsCode} language="python" title="association_metrics.py" />
        </div>
      </section>

      {/* Apriori Algorithm */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <Database size={24} style={{ color: '#8b5cf6' }} /> Apriori Algorithm
        </h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            <strong className="text-white">Apriori</strong> uses the "apriori principle": 
            <em className="text-purple-400"> if an itemset is infrequent, all its supersets are also infrequent</em>. 
            This dramatically prunes the search space.
          </p>

          <FlowDiagram chart={aprioriDiagram} title="Apriori Algorithm Flow" />

          <div className="mt-6 p-4 rounded-lg" style={{ background: 'rgba(139, 92, 246, 0.1)', borderLeft: '4px solid #8b5cf6' }}>
            <h4 className="font-semibold text-purple-400 mb-2">Apriori Principle</h4>
            <p className="text-gray-400 text-sm">
              If {"{bread, milk}"} is infrequent (support &lt; threshold), then any superset like 
              {"{bread, milk, butter}"} must also be infrequent. No need to count it!
            </p>
          </div>

          <CodeBlock code={aprioriCode} language="python" title="apriori_algorithm.py" />
        </div>
      </section>

      {/* FP-Growth */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4">🌳 FP-Growth Algorithm</h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            <strong className="text-white">FP-Growth</strong> (Frequent Pattern Growth) is faster than 
            Apriori because it avoids candidate generation. It compresses the database into an 
            FP-tree and mines patterns directly from this structure.
          </p>

          <FlowDiagram chart={fpGrowthDiagram} title="FP-Growth Process" />

          <div className="grid md:grid-cols-2 gap-4 my-6">
            <div className="p-4 rounded-lg" style={{ background: 'rgba(34, 197, 94, 0.1)' }}>
              <h4 className="font-semibold text-green-400 mb-2">✓ Advantages over Apriori</h4>
              <ul className="text-gray-400 text-sm space-y-1">
                <li>• No candidate generation</li>
                <li>• Scans database only twice</li>
                <li>• Faster on large datasets</li>
                <li>• Memory efficient (compressed tree)</li>
              </ul>
            </div>
            <div className="p-4 rounded-lg" style={{ background: 'rgba(239, 68, 68, 0.1)' }}>
              <h4 className="font-semibold text-red-400 mb-2">✗ Limitations</h4>
              <ul className="text-gray-400 text-sm space-y-1">
                <li>• Building FP-tree can be expensive</li>
                <li>• May not fit in memory for huge datasets</li>
                <li>• Tree construction overhead</li>
              </ul>
            </div>
          </div>

          <CodeBlock code={fpGrowthCode} language="python" title="fp_growth.py" />
        </div>
      </section>

      {/* Sequential Patterns */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4">📊 Sequential Pattern Mining</h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            <strong className="text-white">Sequential pattern mining</strong> finds patterns in 
            <em className="text-blue-400"> ordered sequences</em> of events or items. Unlike regular 
            association rules, the order matters!
          </p>

          <div className="p-4 rounded-lg mb-6" style={{ background: 'rgba(59, 130, 246, 0.1)' }}>
            <h4 className="font-semibold text-blue-400 mb-2">Examples</h4>
            <ul className="text-gray-400 text-sm space-y-1">
              <li>• Customer purchases over time: TV → Sound System → Gaming Console</li>
              <li>• Website navigation: Home → Products → Cart → Checkout</li>
              <li>• Medical treatments: Diagnosis → Medication → Follow-up</li>
            </ul>
          </div>

          <CodeBlock code={sequentialPatternCode} language="python" title="sequential_patterns.py" />
        </div>
      </section>

      {/* Real-World Application */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4">🏪 Real-World Application</h2>
        <div className="card">
          <CodeBlock code={realWorldCode} language="python" title="market_basket_report.py" />
        </div>
      </section>

      {/* Key Takeaways */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-white mb-4">📌 Key Takeaways</h2>
        <div 
          className="p-6 rounded-lg"
          style={{ 
            background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(34, 197, 94, 0.1))',
            border: '1px solid rgba(59, 130, 246, 0.3)'
          }}
        >
          <ul className="space-y-3 text-gray-300">
            <li className="flex items-start gap-3">
              <span className="text-blue-400 font-bold">1.</span>
              <span><strong className="text-white">Association rules</strong> discover relationships: "If X, then Y"</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-blue-400 font-bold">2.</span>
              <span><strong className="text-white">Support</strong> = frequency, <strong className="text-white">Confidence</strong> = rule strength, <strong className="text-white">Lift</strong> = interestingness</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-blue-400 font-bold">3.</span>
              <span><strong className="text-white">Apriori</strong> uses bottom-up candidate generation with pruning</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-blue-400 font-bold">4.</span>
              <span><strong className="text-white">FP-Growth</strong> is faster (no candidate generation) using an FP-tree</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-blue-400 font-bold">5.</span>
              <span><strong className="text-white">Sequential patterns</strong> find order-dependent relationships</span>
            </li>
          </ul>
        </div>
      </section>

      {/* Navigation */}
      <div className="flex justify-between items-center pt-8 border-t" style={{ borderColor: 'rgba(59, 130, 246, 0.2)' }}>
        <Link
          to="/module/4"
          className="text-gray-400 hover:text-white flex items-center gap-2"
        >
          <span>←</span>
          Previous: Classification Advanced
        </Link>
        <Link
          to="/module/6"
          className="btn-primary flex items-center gap-2"
        >
          Next: Cluster Analysis
          <span>→</span>
        </Link>
      </div>
    </div>
  );
}
