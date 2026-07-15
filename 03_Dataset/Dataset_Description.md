# Dataset Description - Crop Recommendation Dataset

## Overview
The Crop Recommendation dataset is designed to help farmers and agricultural researchers determine the most suitable crop for specific soil and weather conditions.

## Dataset Details
- **Total Samples**: 2200
- **Total Features**: 7 (input) + 1 (target)
- **Number of Crops**: 22
- **Format**: CSV
- **Missing Values**: None (fully cleaned)

## Feature Descriptions

### 1. N (Nitrogen)
- **Description**: Amount of nitrogen present in the soil
- **Unit**: kg/ha (kilograms per hectare)
- **Range**: 0 to 140
- **Importance**: Nitrogen is essential for plant growth and leaf development

### 2. P (Phosphorus)
- **Description**: Amount of phosphorus present in the soil
- **Unit**: kg/ha (kilograms per hectare)
- **Range**: 5 to 145
- **Importance**: Phosphorus is crucial for root development and flowering

### 3. K (Potassium)
- **Description**: Amount of potassium present in the soil
- **Unit**: kg/ha (kilograms per hectare)
- **Range**: 5 to 205
- **Importance**: Potassium helps in disease resistance and fruit quality

### 4. Temperature
- **Description**: Average temperature of the region
- **Unit**: Degrees Celsius (°C)
- **Range**: 8 to 45
- **Importance**: Different crops require different temperature ranges

### 5. Humidity
- **Description**: Average humidity of the region
- **Unit**: Percentage (%)
- **Range**: 14 to 100
- **Importance**: Humidity affects crop growth and disease susceptibility

### 6. pH
- **Description**: Soil pH value
- **Unit**: pH scale (0-14)
- **Range**: 3.5 to 9.9
- **Importance**: pH determines nutrient availability in soil

### 7. Rainfall
- **Description**: Average rainfall in the region
- **Unit**: Millimeters (mm)
- **Range**: 20 to 300
- **Importance**: Rainfall is crucial for crop irrigation

### 8. Label (Target)
- **Description**: Recommended crop name
- **Type**: Categorical (String)
- **Unique Values**: 22 different crops

## Crop List
The dataset contains the following 22 crops:

1. Apple
2. Banana
3. Barley
4. Chickpea
5. Coconut
6. Coffee
7. Cotton
8. Grapes
9. Groundnut
10. Jute
11. Lentil
12. Maize
13. Mango
14. Mothbeans
15. Mungbean
16. Muskmelon
17. Orange
18. Papaya
19. Pigeonpeas
20. Rice
21. Sugarcane
22. Watermelon

## Statistical Summary

### Feature Statistics

| Feature | Mean | Std | Min | 25% | 50% | 75% | Max |
|---------|------|-----|-----|-----|-----|-----|-----|
| N | 50.55 | 36.92 | 0 | 21 | 43 | 80 | 140 |
| P | 53.36 | 32.99 | 5 | 28 | 51 | 78 | 145 |
| K | 48.15 | 48.26 | 5 | 20 | 32 | 70 | 205 |
| Temperature | 25.62 | 5.48 | 8.83 | 22.77 | 25.76 | 28.62 | 43.68 |
| Humidity | 71.48 | 22.26 | 14.26 | 60.66 | 72.93 | 82.04 | 99.98 |
| pH | 6.47 | 0.77 | 3.50 | 5.97 | 6.43 | 7.00 | 9.94 |
| Rainfall | 103.46 | 54.96 | 20.21 | 54.00 | 106.61 | 142.51 | 298.56 |

## Data Quality
- **Missing Values**: None
- **Duplicates**: None
- **Outliers**: Handled during preprocessing
- **Data Type**: All numeric features are float/int, label is string

## Applications
1. Crop recommendation for farmers
2. Agricultural research and planning
3. Policy making for agriculture
4. Smart farming systems
5. Yield prediction systems
