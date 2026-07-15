

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set FiveThirtyEight style
plt.style.use('fivethirtyeight')
sns.set_palette("husl")

print("="*70)
print("📊 TASK 3: Univariate Analysis")
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

# ============================================================
# 1. Histograms with KDE (Distribution Analysis)
# ============================================================

print("\n📊 Creating histograms with KDE...")

numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
fig.suptitle('Univariate Analysis - Feature Distributions', fontsize=16, fontweight='bold')

for i, col in enumerate(numeric_cols):
    row, col_idx = i // 4, i % 4
    
    # Histogram with KDE
    sns.histplot(df[col], kde=True, ax=axes[row, col_idx], color='blue', alpha=0.6)
    axes[row, col_idx].set_title(f'Distribution of {col}')
    axes[row, col_idx].set_xlabel(col)
    axes[row, col_idx].set_ylabel('Frequency')
    
    # Add mean and median lines
    mean_val = df[col].mean()
    median_val = df[col].median()
    axes[row, col_idx].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    axes[row, col_idx].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
    axes[row, col_idx].legend(fontsize=8)

# Remove empty subplot
if len(numeric_cols) < 8:
    axes[1, 3].remove()

plt.tight_layout()
plt.savefig('plots/univariate_histograms.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Histograms saved to: plots/univariate_histograms.png")

# ============================================================
# 2. Boxplots (Outlier Detection)
# ============================================================

print("\n📊 Creating boxplots for outlier detection...")

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
fig.suptitle('Univariate Analysis - Boxplots for Outlier Detection', fontsize=16, fontweight='bold')

for i, col in enumerate(numeric_cols):
    row, col_idx = i // 4, i % 4
    sns.boxplot(y=df[col], ax=axes[row, col_idx], color='skyblue')
    axes[row, col_idx].set_title(f'Boxplot of {col}')
    axes[row, col_idx].set_ylabel(col)

if len(numeric_cols) < 8:
    axes[1, 3].remove()

plt.tight_layout()
plt.savefig('plots/univariate_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Boxplots saved to: plots/univariate_boxplots.png")

# ============================================================
# 3. Crop Distribution (Countplot)
# ============================================================

print("\n📊 Creating crop distribution chart...")

plt.figure(figsize=(14, 8))
crop_counts = df['label'].value_counts()
sns.barplot(x=crop_counts.values, y=crop_counts.index, palette='viridis')
plt.title('Univariate Analysis - Crop Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Count')
plt.ylabel('Crop')
plt.tight_layout()
plt.savefig('plots/univariate_crop_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Crop distribution saved to: plots/univariate_crop_distribution.png")

# ============================================================
# 4. Statistical Summary
# ============================================================

print("\n📊 Statistical Summary:")
print(df[numeric_cols].describe())

# ============================================================
# 5. Skewness Analysis
# ============================================================

print("\n📊 Skewness of Features:")
for col in numeric_cols:
    skewness = df[col].skew()
    if abs(skewness) < 0.5:
        status = "Symmetric"
    elif abs(skewness) < 1:
        status = "Moderately skewed"
    else:
        status = "Highly skewed"
    print(f"  {col}: {skewness:.3f} ({status})")

# ============================================================
# 6. Outlier Detection Summary
# ============================================================

print("\n🔍 Outlier Detection (IQR Method):")

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers = len(df[(df[col] < lower) | (df[col] > upper)])
    print(f"  {col}: {outliers} outliers found")

print("\n" + "="*70)
print("✅ TASK 3 COMPLETE!")
print("="*70)