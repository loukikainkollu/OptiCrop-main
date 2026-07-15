

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set FiveThirtyEight style
plt.style.use('fivethirtyeight')
sns.set_palette("husl")

print("="*70)
print("📊 TASK 4: Bivariate Analysis")
print("="*70)


# ============================================================
# Load Dataset
# ============================================================

print("\n📂 Loading dataset...")

dataset_path = '../03_Dataset/Crop_recommendation.csv'

if not os.path.exists(dataset_path):
    print(f"❌ Dataset not found at: {dataset_path}")
    exit()

df = pd.read_csv(dataset_path)
print(f"✅ Dataset loaded: {df.shape[0]} rows")

# Create plots directory
os.makedirs('plots', exist_ok=True)

numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# ============================================================
# 1. Scatter Plots - Feature Relationships
# ============================================================

print("\n📊 Creating scatter plots...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Bivariate Analysis - Feature Relationships', fontsize=16, fontweight='bold')

scatter_pairs = [
    ('temperature', 'humidity'),
    ('temperature', 'rainfall'),
    ('humidity', 'rainfall'),
    ('N', 'P'),
    ('N', 'K'),
    ('P', 'K')
]

for i, (x, y) in enumerate(scatter_pairs):
    row, col_idx = i // 3, i % 3
    sns.scatterplot(data=df, x=x, y=y, ax=axes[row, col_idx], alpha=0.6, hue='label', legend=False)
    axes[row, col_idx].set_title(f'{x} vs {y}')
    axes[row, col_idx].set_xlabel(x)
    axes[row, col_idx].set_ylabel(y)

plt.tight_layout()
plt.savefig('plots/bivariate_scatter_plots.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Scatter plots saved to: plots/bivariate_scatter_plots.png")

# ============================================================
# 2. Crop vs Feature Boxplots
# ============================================================

print("\n📊 Creating crop vs feature boxplots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Bivariate Analysis - Crop vs Features', fontsize=16, fontweight='bold')

bivariate_features = ['temperature', 'humidity', 'ph', 'rainfall']

for i, feature in enumerate(bivariate_features):
    row, col_idx = i // 2, i % 2
    # Get top 10 crops for this feature
    top_crops = df.groupby('label')[feature].mean().nlargest(10).index
    subset = df[df['label'].isin(top_crops)]
    sns.boxplot(data=subset, x='label', y=feature, ax=axes[row, col_idx], palette='Set2')
    axes[row, col_idx].set_title(f'{feature} by Top 10 Crops')
    axes[row, col_idx].set_xlabel('Crop')
    axes[row, col_idx].set_ylabel(feature)
    axes[row, col_idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('plots/bivariate_crop_features.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Crop-feature boxplots saved to: plots/bivariate_crop_features.png")

# ============================================================
# 3. Correlation Analysis
# ============================================================

print("\n📊 Correlation Analysis:")

correlation_matrix = df[numeric_cols].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix.round(2))

# Find strongest correlations
corr_pairs = correlation_matrix.unstack().sort_values(ascending=False)
corr_pairs = corr_pairs[corr_pairs < 1]  # Remove self-correlations

print("\n🔗 Strongest Positive Correlations:")
for pair in corr_pairs.head(5).items():
    print(f"  {pair[0][0]} ↔ {pair[0][1]}: {pair[1]:.2f}")

print("\n🔗 Strongest Negative Correlations:")
for pair in corr_pairs.tail(5).items():
    print(f"  {pair[0][0]} ↔ {pair[0][1]}: {pair[1]:.2f}")

# ============================================================
# 4. Save Correlation Matrix
# ============================================================

print("\n💾 Saving correlation matrix...")
correlation_matrix.to_csv('correlation_matrix.csv')
print("✅ Correlation matrix saved to: correlation_matrix.csv")

print("\n" + "="*70)
print("✅ TASK 4 COMPLETE!")
print("="*70)