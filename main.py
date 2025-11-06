# L1 vs L2 Regularization using Logistic Regression (Wine Dataset)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = load_wine()
X, y = data.data, data.target
feature_names = data.feature_names

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression with L1 and L2
logreg_l1 = LogisticRegression(penalty='l1', solver='saga', multi_class='ovr', max_iter=5000, random_state=42)
logreg_l2 = LogisticRegression(penalty='l2', solver='saga', multi_class='ovr', max_iter=5000, random_state=42)

logreg_l1.fit(X_train, y_train)
logreg_l2.fit(X_train, y_train)

# Model accuracy
acc_l1 = accuracy_score(y_test, logreg_l1.predict(X_test))
acc_l2 = accuracy_score(y_test, logreg_l2.predict(X_test))

print("=== L1 vs L2 Regularization on Wine Dataset ===")
print(f"L1 Accuracy: {acc_l1:.4f}")
print(f"L2 Accuracy: {acc_l2:.4f}")

# Create dataframe of weights
df = pd.DataFrame({
    'Feature': feature_names,
    'L1 Weights (Class 0)': logreg_l1.coef_[0],
    'L2 Weights (Class 0)': logreg_l2.coef_[0]
})

# Count near-zero weights
l1_zeros = np.sum(np.abs(logreg_l1.coef_) < 0.001)
l2_zeros = np.sum(np.abs(logreg_l2.coef_) < 0.001)

print(f"L1 Near-zero weights: {l1_zeros}")
print(f"L2 Near-zero weights: {l2_zeros}")

# --- Visualization 1: Bar chart for L1 vs L2 weights ---
plt.figure(figsize=(10, 6))
x = np.arange(len(feature_names))
width = 0.35
plt.bar(x - width/2, logreg_l1.coef_[0], width, label='L1', color='#e74c3c', alpha=0.8)
plt.bar(x + width/2, logreg_l2.coef_[0], width, label='L2', color='#3498db', alpha=0.8)
plt.xlabel('Features')
plt.ylabel('Weight Value')
plt.title('L1 vs L2 Regularization - Weight Comparison (Class 0)')
plt.xticks(x, feature_names, rotation=45, ha='right')
plt.axhline(y=0, color='black', linewidth=1)
plt.legend()
plt.tight_layout()
plt.show()

# --- Visualization 2: Histogram of weight distributions ---
plt.figure(figsize=(8, 5))
plt.hist(logreg_l1.coef_[0], bins=10, alpha=0.7, label='L1', color='#e74c3c')
plt.hist(logreg_l2.coef_[0], bins=10, alpha=0.7, label='L2', color='#3498db')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.title('Weight Distribution - L1 vs L2 Regularization')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.legend()
plt.tight_layout()
plt.show()

print("✓ Comparison and visualization completed successfully.")
