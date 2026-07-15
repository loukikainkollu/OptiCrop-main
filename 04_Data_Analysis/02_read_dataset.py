
import pandas as pd
import numpy as np
import os

print("="*70)
print("📊 TASK 2: Reading the Dataset")
print("="*70)

# ============================================================
# Load Dataset
# ============================================================

print("\n📂 Loading dataset...")

# Check if dataset exists
dataset_path = '../03_Dataset/Crop_recommendation.csv'

if not os.path.exists(dataset_path):
    print(f"❌ Dataset not found at: {dataset_path}")
    print("Please ensure Crop_recommendation.csv is in 03_Dataset folder")
    exit()

# Load dataset
df = pd.read_csv(dataset_path)

print(f"✅ Dataset loaded successfully!")
print(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   Columns: {list(df.columns)}")
print(f"   Unique crops: {df['label'].nunique()}")

# ============================================================
# Display First 5 Rows (head())
# ============================================================

print("\n📋 First 5 rows (head()):")
print(df.head())

# ============================================================
# Dataset Information
# ============================================================

print("\n📋 Dataset Info:")
print(df.info())

# ============================================================
# Statistical Summary
# ============================================================

print("\n📊 Statistical Summary:")
print(df.describe())

# ============================================================
# Check for Missing Values
# ============================================================

print("\n🔍 Missing Values:")
print(df.isnull().sum())

# ============================================================
# Check for Duplicates
# ============================================================

print(f"\n🔍 Duplicate Rows: {df.duplicated().sum()}")

# ============================================================
# Crop Distribution
# ============================================================

print("\n🌾 Crop Distribution:")
crop_counts = df['label'].value_counts()
print(crop_counts)

print(f"\nMost common crop: {crop_counts.index[0]} ({crop_counts.values[0]} samples)")
print(f"Least common crop: {crop_counts.index[-1]} ({crop_counts.values[-1]} samples)")

# ============================================================
# Save Dataset Info
# ============================================================

print("\n💾 Saving dataset information...")

with open('dataset_info.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("DATASET INFORMATION - OPTICROP\n")
    f.write("="*70 + "\n\n")
    f.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    f.write(f"Columns: {list(df.columns)}\n")
    f.write(f"Unique crops: {df['label'].nunique()}\n\n")
    f.write("Feature Ranges:\n")
    for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
        f.write(f"  {col}: {df[col].min():.2f} - {df[col].max():.2f}\n")

print("✅ Dataset info saved to: dataset_info.txt")

print("\n" + "="*70)
print("✅ TASK 2 COMPLETE!")
print("="*70)