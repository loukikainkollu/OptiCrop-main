# Data Analysis - OptiCrop Project

## Overview
This folder contains comprehensive Exploratory Data Analysis (EDA) performed on the agricultural dataset using Python scripts in VS Code.

## Dataset Information
- **Name**: Crop Recommendation Dataset
- **Source**: Kaggle - Smart Agricultural Production Optimizing Engine
- **Link**: https://www.kaggle.com/datasets/chitrakumari25/smart-agricultural-production-optimizing-engine
- **Shape**: 2200 rows, 8 columns
- **Features**: N, P, K, Temperature, Humidity, pH, Rainfall
- **Target**: Crop Label (22 different crops)

## Files in this Folder

| File | Description |
|------|-------------|
| README.md | This file - overview of analysis |
| 01_import_libraries.py | Import all required libraries |
| 02_read_dataset.py | Load and explore dataset |
| 03_univariate_analysis.py | Single variable analysis |
| 04_bivariate_analysis.py | Two variable analysis |
| 05_multivariate_analysis.py | Multiple variable analysis |
| 06_complete_eda.py | Complete EDA in one script |
| plots/ | All visualization files |

## How to Run

```bash
# Run individual scripts
python 01_import_libraries.py
python 02_read_dataset.py
python 03_univariate_analysis.py
python 04_bivariate_analysis.py
python 05_multivariate_analysis.py
