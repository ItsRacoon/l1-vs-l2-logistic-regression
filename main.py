import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# L1 Regularization (Lasso)
logreg_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
logreg_l1.fit(X_train, y_train)

# L2 Regularization (Ridge)
logreg_l2 = LogisticRegression(penalty='l2', solver='liblinear', C=1.0)
logreg_l2.fit(X_train, y_train)


df = pd.DataFrame({
    'Feature': feature_names,
    'L1 Coefficient': logreg_l1.coef_[0],
    'L2 Coefficient': logreg_l2.coef_[0]
})

print("\nCoefficient Comparison:\n")
print(df)


acc_l1 = accuracy_score(y_test, logreg_l1.predict(X_test))
acc_l2 = accuracy_score(y_test, logreg_l2.predict(X_test))

print(f"\nL1 Accuracy: {acc_l1:.4f}")
print(f"L2 Accuracy: {acc_l2:.4f}")


plt.figure(figsize=(12,6))
plt.plot(logreg_l1.coef_[0], 'o', label='L1')
plt.plot(logreg_l2.coef_[0], 'x', label='L2')
plt.xticks(np.arange(len(feature_names)), feature_names, rotation=90)
plt.ylabel("Coefficient Value")
plt.title("L1 vs L2 Coefficients - Breast Cancer Dataset")
plt.legend()
plt.tight_layout()
plt.show()
