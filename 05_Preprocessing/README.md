# Data Preprocessing - OptiCrop Project

## Overview
This folder contains all preprocessing steps performed on the agricultural dataset before model building.

## Dataset Information
- **Name**: Crop Recommendation Dataset
- **Source**: Kaggle - Smart Agricultural Production Optimizing Engine
- **Shape**: 2200 rows, 8 columns
- **Features**: N, P, K, Temperature, Humidity, pH, Rainfall
- **Target**: Crop Label (22 different crops)

## Preprocessing Tasks Performed

### Task 1: Handling Outliers
- **Method**: IQR (Interquartile Range) with Winsorization
- **Result**: No data loss, outliers capped
- **Features handled**: N, P, K, Temperature, Humidity, pH, Rainfall

### Task 2: Missing Value Handling
- **Method**: Checked for null values
- **Result**: No missing values found in dataset

### Task 3: Feature Engineering
- **Seasonal Crop Categorization**: Grouped crops by season
- **Feature Scaling**: StandardScaler applied
- **Label Encoding**: Crop names converted to numerical values

### Task 4: Data Splitting
- **Split**: 80% Training, 20% Testing
- **Method**: Stratified split (preserves crop distribution)
- **Random State**: 42 (for reproducibility)

## Files in this Folder

| File | Description |
|------|-------------|
| README.md | This file - overview of preprocessing |
| preprocessing.py | Main preprocessing script |
| Missing_Value_Handling.md | Missing value analysis |
| Outlier_Detection.md | Outlier handling methodology |
| Feature_Engineering.md | Feature engineering details |

## Output Files

| File | Description |
|------|-------------|
| X_train.npy | Training features (80%) |
| X_test.npy | Testing features (20%) |
| y_train.npy | Training labels |
| y_test.npy | Testing labels |
| scaler.pkl | StandardScaler for input scaling |

## How to Run

```bash
# Navigate to preprocessing folder
cd 05_Preprocessing

# Run the script
python preprocessing.py