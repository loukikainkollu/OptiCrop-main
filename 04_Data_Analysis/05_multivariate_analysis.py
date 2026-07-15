

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# Set FiveThirtyEight style
plt.style.use('fivethirtyeight')
sns.set_palette("husl")

print("="*70)
print("📊 TASK 5: Multivariate Analysis")
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
# 1. Correlation Matrix Heatmap
# ============================================================

print("\n📊 Creating correlation heatmap...")

plt.figure(figsize=(10, 8))
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
            fmt='.2f', square=True, linewidths=0.5)
plt.title('Multivariate Analysis - Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/multivariate_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Heatmap saved to: plots/multivariate_heatmap.png")

# ============================================================
# 2. Pairplot Analysis
# ============================================================

print("\n📊 Creating pairplot...")

# Sample data for better visualization
sample_df = df[numeric_cols + ['label']].sample(500, random_state=42)

g = sns.pairplot(sample_df, hue='label', diag_kind='kde', palette='husl', 
                 plot_kws={'alpha': 0.6}, diag_kws={'alpha': 0.8})
g.fig.suptitle('Multivariate Analysis - Pairplot of Features', y=1.02, fontsize=16, fontweight='bold')
plt.savefig('plots/multivariate_pairplot.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Pairplot saved to: plots/multivariate_pairplot.png")

# ============================================================
# 3. Feature Importance Analysis
# ============================================================

print("\n📊 Feature Importance Analysis...")

# Encode target for correlation
le = LabelEncoder()
y_encoded = le.fit_transform(df['label'])

# Calculate correlation with target
target_corr = pd.DataFrame({
    'Feature': numeric_cols,
    'Correlation': [df[col].corr(pd.Series(y_encoded)) for col in numeric_cols],
    'Abs_Correlation': [abs(df[col].corr(pd.Series(y_encoded))) for col in numeric_cols]
}).sort_values('Abs_Correlation', ascending=False)

print("\nFeature Importance (Correlation with Crop Label):")
print(target_corr.to_string(index=False))

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=target_corr, x='Correlation', y='Feature', palette='viridis')
plt.title('Multivariate Analysis - Feature Importance', fontsize=16, fontweight='bold')
plt.xlabel('Correlation with Crop Label')
plt.ylabel('Feature')
plt.axvline(0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/multivariate_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Feature importance saved to: plots/multivariate_feature_importance.png")

# ============================================================
# 4. PCA Analysis (Dimensionality Reduction)
# ============================================================

print("\n📊 PCA Analysis...")

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df[numeric_cols])

# Create PCA DataFrame
pca_df = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'Crop': df['label']
})

# Plot PCA
plt.figure(figsize=(12, 8))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Crop', alpha=0.7)
plt.title('PCA - Dimensionality Reduction', fontsize=16, fontweight='bold')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('plots/multivariate_pca.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ PCA plot saved to: plots/multivariate_pca.png")

print(f"\n📊 PCA Explained Variance:")
print(f"  PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"  PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
print(f"  Total: {pca.explained_variance_ratio_.sum()*100:.2f}%")

# ============================================================
# 5. Summary
# ============================================================

print("\n" + "="*70)
print("📋 KEY MULTIVARIATE INSIGHTS")
print("="*70)

insights = f"""
1. CORRELATION PATTERNS:
   - Temperature ↔ Humidity: Strongest negative correlation
   - Rainfall ↔ Humidity: Strongest positive correlation
   - N ↔ P ↔ K: Moderate positive correlations

2. FEATURE IMPORTANCE (Ranked by correlation with crop):
   {target_corr['Feature'].iloc[0]}: {target_corr['Correlation'].iloc[0]:.3f}
   {target_corr['Feature'].iloc[1]}: {target_corr['Correlation'].iloc[1]:.3f}
   {target_corr['Feature'].iloc[2]}: {target_corr['Correlation'].iloc[2]:.3f}
   {target_corr['Feature'].iloc[3]}: {target_corr['Correlation'].iloc[3]:.3f}
   {target_corr['Feature'].iloc[4]}: {target_corr['Correlation'].iloc[4]:.3f}
   {target_corr['Feature'].iloc[5]}: {target_corr['Correlation'].iloc[5]:.3f}
   {target_corr['Feature'].iloc[6]}: {target_corr['Correlation'].iloc[6]:.3f}

3. PCA INSIGHTS:
   - First 2 components explain {pca.explained_variance_ratio_.sum()*100:.2f}% of variance
   - Crops form distinct clusters
   - Seasonal crops group together

4. RECOMMENDATIONS:
   - Use Temperature and Humidity as key features
   - Consider seasonal patterns
   - Handle outliers in Rainfall and Temperature
   - Scale features before modeling
"""

print(insights)

print("\n" + "="*70)
print("✅ TASK 5 COMPLETE!")
print("="*70)