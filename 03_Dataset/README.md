# Dataset - OptiCrop Project

## Overview
This folder contains the agricultural dataset used for crop recommendation in the OptiCrop project.

## Dataset Information
- **Dataset Name**: Crop Recommendation Dataset
- **Source**: Kaggle - Smart Agricultural Production Optimizing Engine
- **Link**: https://www.kaggle.com/datasets/chitrakumari25/smart-agricultural-production-optimizing-engine
- **File**: Crop_recommendation.csv
- **Format**: CSV (Comma Separated Values)

## Dataset Description
This dataset contains soil and weather parameters for various crops, used to build a crop recommendation system.

### Features (Input Variables)
| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| N | Nitrogen content in soil | kg/ha | 0-140 |
| P | Phosphorus content in soil | kg/ha | 5-145 |
| K | Potassium content in soil | kg/ha | 5-205 |
| temperature | Temperature | °C | 8-45 |
| humidity | Humidity | % | 14-100 |
| ph | Soil pH value | - | 3.5-9.9 |
| rainfall | Rainfall | mm | 20-300 |

### Target Variable
| Feature | Description |
|---------|-------------|
| label | Crop name to be recommended |

## Files in this Folder
| File | Description |
|------|-------------|
| Crop_recommendation.csv | Main dataset file |
| Dataset_Description.md | Detailed dataset description |
| Data_Source.md | Data source and reference information |
| README.md | This file |

## Usage
```python
import pandas as pd

# Load dataset
df = pd.read_csv('03_Dataset/Crop_recommendation.csv')

# View data
print(df.head())
print(df.info())