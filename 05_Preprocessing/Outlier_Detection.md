# Outlier Detection and Handling - OptiCrop Project

## Overview
This document describes the outlier detection and handling methodology used for the crop recommendation dataset.

## Methodology

### IQR Method
The Interquartile Range (IQR) method was used to detect outliers:

1. **Calculate Quartiles**:
   - Q1 (25th percentile)
   - Q3 (75th percentile)

2. **Calculate IQR**:
   IQR = Q3 - Q1

3. **Define Bounds**:
   Lower Bound = Q1 - 1.5 * IQR
   Upper Bound = Q3 + 1.5 * IQR

4. **Detect Outliers**:
   - Values below Lower Bound are outliers
   - Values above Upper Bound are outliers

### Winsorization
Instead of removing outliers, we used **Winsorization** (capping):
- Values below Lower Bound are set to Lower Bound
- Values above Upper Bound are set to Upper Bound

**Advantages**:
- Preserves all data points
- Reduces impact of extreme values
- Maintains dataset size

## Results

### Outlier Summary
| Feature | Outliers Before | Lower Bound | Upper Bound |
|---------|----------------|-------------|-------------|
| N | 0 | - | - |
| P | 138 | -19.00 | 115.00 |
| K | 200 | -33.50 | 102.50 |
| Temperature | 86 | 13.96 | 36.17 |
| Humidity | 30 | 27.03 | 122.61 |
| pH | 57 | 4.59 | 9.29 |
| Rainfall | 100 | 14.24 | 174.58 |

### Features with Outliers
1. **P (Phosphorus)**: 138 outliers
2. **K (Potassium)**: 200 outliers
3. **Temperature**: 86 outliers
4. **Humidity**: 30 outliers
5. **pH**: 57 outliers
6. **Rainfall**: 100 outliers
7. **N (Nitrogen)**: 0 outliers

## Why This Approach?
- Dataset had minimal outliers overall
- Winsorization preserves all data points
- Suitable for machine learning models
- Prevents loss of valuable agricultural data
- Maintains data distribution shape

## Code Used
```python
def handle_outliers(df):
    df_clean = df.copy()
    numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        # Cap outliers (Winsorization)
        df_clean[col] = df[col].clip(lower=lower, upper=upper)
    
    return df_clean