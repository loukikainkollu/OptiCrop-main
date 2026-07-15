
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("="*70)
print("🌾 DATA PREPROCESSING - OPTICROP")
print("="*70)

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

print("\n📊 STEP 1: Loading Dataset")

dataset_path = '../03_Dataset/Crop_recommendation.csv'

if not os.path.exists(dataset_path):
    print(f"❌ Dataset not found at: {dataset_path}")
    print("Please ensure Crop_recommendation.csv is in 03_Dataset folder")
    exit()

df = pd.read_csv(dataset_path)
print(f"✅ Dataset loaded successfully!")
print(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   Columns: {list(df.columns)}")
print(f"   Unique crops: {df['label'].nunique()}")

print("\n📋 First 5 rows:")
print(df.head())

# ============================================================
# STEP 2: CHECK MISSING VALUES
# ============================================================

print("\n🔍 STEP 2: Checking for Missing Values")

missing_values = df.isnull().sum()
print("\nMissing Values Summary:")
print(missing_values)

if missing_values.sum() == 0:
    print("\n✅ No missing values found in the dataset!")
else:
    print("\n⚠️ Missing values found. Handling them...")
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)

print("\n✅ Missing value check complete!")

# ============================================================
# STEP 3: HANDLING OUTLIERS
# ============================================================

print("\n🔍 STEP 3: Handling Outliers")

def handle_outliers(df):
    df_clean = df.copy()
    numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    print("\nOutlier Analysis (IQR Method):")
    outlier_summary = []
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers_before = len(df[(df[col] < lower) | (df[col] > upper)])
        print(f"  {col}: {outliers_before} outliers before handling")
        
        df_clean[col] = df[col].clip(lower=lower, upper=upper)
        
        outlier_summary.append({
            'Feature': col,
            'Outliers_Before': outliers_before,
            'Lower_Bound': round(lower, 2),
            'Upper_Bound': round(upper, 2)
        })
    
    return df_clean, outlier_summary

df_clean, outlier_summary = handle_outliers(df)

outlier_df = pd.DataFrame(outlier_summary)
print("\n📊 Outlier Summary:")
print(outlier_df.to_string(index=False))

print(f"\n✅ Outliers handled. Data shape: {df_clean.shape}")

# ============================================================
# STEP 4: FEATURE ENGINEERING - SEASONAL CROPS
# ============================================================

print("\n🌱 STEP 4: Feature Engineering - Seasonal Crops")

season_mapping = {
    'rice': 'Kharif (Monsoon)',
    'maize': 'Kharif (Monsoon)',
    'cotton': 'Kharif (Monsoon)',
    'groundnut': 'Kharif (Monsoon)',
    'pigeonpeas': 'Kharif (Monsoon)',
    'mungbean': 'Kharif (Monsoon)',
    'blackgram': 'Kharif (Monsoon)',
    'soybean': 'Kharif (Monsoon)',
    'sesame': 'Kharif (Monsoon)',
    'wheat': 'Rabi (Winter)',
    'chickpea': 'Rabi (Winter)',
    'lentil': 'Rabi (Winter)',
    'mustard': 'Rabi (Winter)',
    'barley': 'Rabi (Winter)',
    'peas': 'Rabi (Winter)',
    'sugarcane': 'Perennial',
    'coffee': 'Perennial',
    'tea': 'Perennial',
    'coconut': 'Perennial',
    'banana': 'Perennial',
    'orange': 'Perennial',
    'apple': 'Perennial',
    'mango': 'Perennial',
    'grapes': 'Perennial',
    'watermelon': 'Summer',
    'muskmelon': 'Summer',
    'pomegranate': 'Summer',
    'papaya': 'Perennial'
}

df_clean['Season'] = df_clean['label'].map(season_mapping)

print("\nSeasonal Crop Distribution:")
season_counts = df_clean['Season'].value_counts()

for season, count in season_counts.items():
    crops = df_clean[df_clean['Season'] == season]['label'].unique()
    print(f"  {season}: {count} samples")
    print(f"    Crops: {', '.join(crops[:5])}{'...' if len(crops)>5 else ''}")

seasonal_nutrients = df_clean.groupby('Season')[['N', 'P', 'K']].mean().round(2)
print("\n📊 Average Nutrient Requirements by Season:")
print(seasonal_nutrients)

# ============================================================
# STEP 5: SPLITTING DATA
# ============================================================

print("\n📊 STEP 5: Splitting Data")

X = df_clean[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df_clean['label']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

le = LabelEncoder()
y_encoded = le.fit_transform(y)

os.makedirs('../06_Model/models', exist_ok=True)
joblib.dump(le, '../06_Model/models/label_encoder.pkl')
print(f"✅ Label encoder saved. Number of crops: {len(le.classes_)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, 'scaler.pkl')
print(f"✅ Scaler saved.")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nData Split Summary:")
print(f"  Total samples: {len(X)}")
print(f"  Training: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Testing: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print(f"\n✅ Preprocessed data saved:")
print(f"  - X_train.npy, X_test.npy, y_train.npy, y_test.npy")
print(f"  - scaler.pkl")

print("\n" + "="*70)
print("✅ DATA PREPROCESSING COMPLETE!")
print("="*70)