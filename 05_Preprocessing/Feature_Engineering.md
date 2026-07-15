# Feature Engineering

## Overview
This document describes the feature engineering steps performed on the crop recommendation dataset.

## Feature Engineering Steps

### 1. Seasonal Crop Categorization
Crops were categorized into seasons for better analysis:

#### Kharif (Monsoon) Season
- Rice, Maize, Cotton, Groundnut, Pigeonpeas
- Mungbean, Blackgram, Soybean, Sesame

#### Rabi (Winter) Season
- Wheat, Chickpea, Lentil, Mustard, Barley, Peas

#### Perennial Crops
- Sugarcane, Coffee, Tea, Coconut, Banana
- Orange, Apple, Mango, Grapes, Papaya

#### Summer Crops
- Watermelon, Muskmelon, Pomegranate

### 2. Feature Scaling
StandardScaler was applied to normalize all features:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)