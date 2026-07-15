import numpy as np
import joblib
import os

def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    """
    Predict best crop based on soil and weather parameters
    
    Parameters:
    - n: Nitrogen content (0-140)
    - p: Phosphorus content (5-145)
    - k: Potassium content (5-205)
    - temperature: Temperature in Celsius (8-45)
    - humidity: Humidity percentage (14-100)
    - ph: pH value (3.5-9.9)
    - rainfall: Rainfall in mm (20-300)
    
    Returns:
    - dict with best_crop, confidence, top_3 recommendations
    """
    # Check if model exists
    if not os.path.exists('models/crop_model.pkl'):
        print("ERROR: crop_model.pkl not found! Please run train.py first.")
        return None
    
    # Load model and preprocessors
    model = joblib.load('models/crop_model.pkl')
    scaler = joblib.load('../05_Preprocessing/scaler.pkl')
    le = joblib.load('models/label_encoder.pkl')
    
    # Prepare input
    input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)
    
    # Get predictions
    probabilities = model.predict_proba(input_scaled)[0]
    
    # Top 3 crops
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    top_3_crops = le.inverse_transform(top_3_indices)
    top_3_confidences = probabilities[top_3_indices]
    
    # Best crop
    best_index = np.argmax(probabilities)
    best_crop = le.inverse_transform([best_index])[0]
    confidence = probabilities[best_index]
    
    return {
        'best_crop': best_crop,
        'confidence': confidence * 100,
        'top_3': list(zip(top_3_crops, top_3_confidences * 100))
    }

if __name__ == "__main__":
    print("="*70)
    print("PREDICTION TEST - OPTICROP")
    print("="*70)
    
    # Test predictions
    test_cases = [
        (90, 42, 43, 21, 82, 6.5, 202),
        (20, 30, 30, 27, 60, 7.2, 50),
        (100, 50, 50, 25, 70, 6.8, 180),
        (80, 60, 70, 28, 75, 6.5, 150)
    ]
    
    print("\nTesting Prediction Function:")
    print("-" * 60)
    
    for i, (n, p, k, temp, hum, ph, rain) in enumerate(test_cases, 1):
        result = predict_crop(n, p, k, temp, hum, ph, rain)
        if result:
            print(f"\nTest {i}:")
            print(f"  Input: N={n}, P={p}, K={k}, Temp={temp}C, Humidity={hum}%, pH={ph}, Rain={rain}mm")
            print(f"  Best Crop: {result['best_crop'].title()}")
            print(f"  Confidence: {result['confidence']:.2f}%")
            print(f"  Top 3: {result['top_3'][0][0].title()} ({result['top_3'][0][1]:.1f}%), "
                  f"{result['top_3'][1][0].title()} ({result['top_3'][1][1]:.1f}%), "
                  f"{result['top_3'][2][0].title()} ({result['top_3'][2][1]:.1f}%)")
    
    print("\n" + "="*70)
    print("PREDICTION TEST COMPLETE!")
    print("="*70)