import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("📊 MODEL EVALUATION - OPTICROP")
print("="*70)

# ============================================================
# LOAD DATA AND MODEL
# ============================================================

print("\n📊 Loading data and model...")

X_test = np.load('../05_Preprocessing/X_test.npy')
y_test = np.load('../05_Preprocessing/y_test.npy')
le = joblib.load('models/label_encoder.pkl')
model = joblib.load('models/crop_model.pkl')

print(f"✅ Data loaded:")
print(f"  Testing samples: {len(X_test)}")
print(f"  Number of crops: {len(le.classes_)}")

# ============================================================
# MAKE PREDICTIONS
# ============================================================

print("\n🤖 Making predictions...")
y_pred = model.predict(X_test)

# ============================================================
# CALCULATE METRICS
# ============================================================

print("\n📊 Calculating metrics...")

accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy*100:.2f}%")

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n📊 Creating confusion matrix...")

plt.figure(figsize=(14, 12))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Crop Recommendation', fontsize=16)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('results/confusion_matrix.png')
plt.close()
print("✅ Confusion matrix saved to results/confusion_matrix.png")

# ============================================================
# SAVE RESULTS
# ============================================================

print("\n💾 Saving results...")

with open('results/evaluation_summary.txt', 'w') as f:
    f.write("MODEL EVALUATION SUMMARY\n")
    f.write("="*50 + "\n")
    f.write(f"Accuracy: {accuracy*100:.2f}%\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred, target_names=le.classes_))

print("✅ Evaluation summary saved to results/evaluation_summary.txt")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*70)
print("📊 EVALUATION SUMMARY")
print("="*70)
print(f"""
✅ Model: {model.__class__.__name__}
✅ Accuracy: {accuracy*100:.2f}%
✅ Confusion Matrix: results/confusion_matrix.png
✅ Classification Report: results/evaluation_summary.txt
""")

print("="*70)
print("✅ MODEL EVALUATION COMPLETE!")
print("="*70)