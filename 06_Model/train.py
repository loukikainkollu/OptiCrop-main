import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("MODEL TRAINING - OPTICROP")
print("="*70)

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# ============================================================
# LOAD PREPROCESSED DATA
# ============================================================

print("\nLoading preprocessed data...")

X_train = np.load('../05_Preprocessing/X_train.npy')
X_test = np.load('../05_Preprocessing/X_test.npy')
y_train = np.load('../05_Preprocessing/y_train.npy')
y_test = np.load('../05_Preprocessing/y_test.npy')

# Load label encoder
le = joblib.load('models/label_encoder.pkl')

print(f"Data loaded:")
print(f"  Training: {len(X_train)} samples")
print(f"  Testing: {len(X_test)} samples")
print(f"  Number of crops: {len(le.classes_)}")

# ============================================================
# K-MEANS CLUSTERING
# ============================================================

print("\nK-Means Clustering")

from sklearn.cluster import KMeans

# Find optimal K using elbow method
inertia = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train)
    inertia.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True, alpha=0.3)
plt.savefig('results/elbow_curve.png')
plt.close()
print("Elbow curve saved to results/elbow_curve.png")

# Apply K-Means with optimal k
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_train)

print(f"K-Means clustering with {optimal_k} clusters complete")

# Save KMeans model
joblib.dump(kmeans, 'models/kmeans_model.pkl')
print("KMeans model saved")

# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("\nLogistic Regression")

logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)

y_pred_logreg = logreg.predict(X_test)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)

print(f"Logistic Regression Accuracy: {accuracy_logreg*100:.2f}%")

joblib.dump(logreg, 'models/logistic_regression_model.pkl')
print("Logistic Regression model saved")

# ============================================================
# MODEL EVALUATION & BEST MODEL
# ============================================================

print("\nModel Evaluation & Best Model")

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42, probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

results = []
model_objects = {}

print("\nTraining and Comparing Models:")
print("-" * 50)

for name, model in models.items():
    print(f"  Training {name}...", end=" ")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append({'Model': name, 'Accuracy': acc})
    model_objects[name] = model
    print(f"OK - Accuracy: {acc*100:.2f}%")

# Find best model
results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
best_model_name = results_df.iloc[0]['Model']
best_model = model_objects[best_model_name]
best_accuracy = results_df.iloc[0]['Accuracy']

print("\n" + "="*50)
print("BEST MODEL RESULTS:")
print(results_df.to_string(index=False))
print("="*50)

# Save best model FIRST before anything else
joblib.dump(best_model, 'models/crop_model.pkl')
print(f"\nBest model saved as models/crop_model.pkl")
print(f"Best Model: {best_model_name} with {best_accuracy*100:.2f}% accuracy")

# Save results
with open('results/accuracy.txt', 'w') as f:
    f.write("MODEL PERFORMANCE RESULTS\n")
    f.write("="*50 + "\n")
    for _, row in results_df.iterrows():
        f.write(f"{row['Model']}: {row['Accuracy']*100:.2f}%\n")
    f.write(f"\nBest Model: {best_model_name} ({best_accuracy*100:.2f}%)\n")

print("Results saved to results/accuracy.txt")

# Confusion Matrix for Best Model
y_pred_best = best_model.predict(X_test)
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('results/confusion_matrix.png')
plt.close()
print("Confusion matrix saved to results/confusion_matrix.png")

# Classification Report
class_report = classification_report(y_test, y_pred_best, target_names=le.classes_)
with open('results/classification_report.txt', 'w') as f:
    f.write(f"CLASSIFICATION REPORT - {best_model_name}\n")
    f.write("="*50 + "\n")
    f.write(class_report)

print("Classification report saved to results/classification_report.txt")

# Feature Importance (for Random Forest)
if hasattr(best_model, 'feature_importances_'):
    feature_names = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
    importances = best_model.feature_importances_
    
    plt.figure(figsize=(10, 6))
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='green')
    plt.xlabel('Importance')
    plt.title('Feature Importance - Random Forest')
    plt.tight_layout()
    plt.savefig('results/feature_importance.png')
    plt.close()
    print("Feature importance saved to results/feature_importance.png")

# Visualize model comparison
plt.figure(figsize=(12, 6))
bars = plt.bar(results_df['Model'], results_df['Accuracy'] * 100, 
               color=['#2ecc71' if m == best_model_name else '#3498db' 
                      for m in results_df['Model']])
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.title('Model Performance Comparison')
plt.xticks(rotation=45, ha='right')
plt.ylim(80, 100)

for bar, acc in zip(bars, results_df['Accuracy'] * 100):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{acc:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('results/model_comparison.png')
plt.close()
print("Model comparison saved to results/model_comparison.png")

print("\n" + "="*70)
print("MODEL TRAINING COMPLETE!")
print("="*70)