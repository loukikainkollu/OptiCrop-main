# Model Building - OptiCrop Project

## Overview
This folder contains all machine learning models built for crop recommendation.

## Models Implemented
1. Logistic Regression
2. Decision Tree
3. Random Forest ⭐ (Best Model)
4. Gradient Boosting
5. SVM
6. KNN
7. Naive Bayes

## Files in this Folder

| File | Description |
|------|-------------|
| README.md | This file - overview of models |
| train.py | Train all models and save best |
| predict.py | Prediction function |
| evaluate.py | Model evaluation and comparison |
| models/ | Saved model files |
| results/ | Model performance results |

## Best Model
**Random Forest** with high accuracy

## How to Run

```bash
# Navigate to model folder
cd 06_Model

# Train models
python train.py

# Make predictions
python predict.py

# Evaluate models
python evaluate.py