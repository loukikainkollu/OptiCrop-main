
print("="*70)
print("📦 TASK 1: Importing Libraries")
print("="*70)

# Core Data Manipulation Libraries
print("\n📊 Loading core libraries...")

import pandas as pd
import numpy as np

print(f"✅ Pandas version: {pd.__version__}")
print(f"✅ NumPy version: {np.__version__}")

# Visualization Libraries (FiveThirtyEight Style)
print("\n📈 Loading visualization libraries...")

import matplotlib.pyplot as plt
import seaborn as sns

# Set FiveThirtyEight style
plt.style.use('fivethirtyeight')
sns.set_palette("husl")

print(f"✅ Matplotlib version: {plt.matplotlib.__version__}")
print(f"✅ Seaborn version: {sns.__version__}")
print(f"✅ Style: FiveThirtyEight (fivethirtyeight)")

# Machine Learning Libraries
print("\n🤖 Loading machine learning libraries...")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix

print(f"✅ Scikit-learn imported successfully")

# Utilities
print("\n🔧 Loading utilities...")

import os
import warnings
warnings.filterwarnings('ignore')

print(f"✅ All libraries imported successfully!")
print("\n" + "="*70)
print("✅ TASK 1 COMPLETE!")
print("="*70)