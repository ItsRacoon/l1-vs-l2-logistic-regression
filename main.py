import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# Load and prepare data
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models with L1 and L2 regularization
logreg_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)
logreg_l1.fit(X_train, y_train)

logreg_l2 = LogisticRegression(penalty='l2', solver='liblinear', C=1.0, random_state=42)
logreg_l2.fit(X_train, y_train)

# Create comparison dataframe
df = pd.DataFrame({
    'Feature': feature_names,
    'L1 Weight': logreg_l1.coef_[0],
    'L2 Weight': logreg_l2.coef_[0]
})

# Calculate metrics
acc_l1 = accuracy_score(y_test, logreg_l1.predict(X_test))
acc_l2 = accuracy_score(y_test, logreg_l2.predict(X_test))

# Print results
print("\n" + "="*70)
print("L1 vs L2 REGULARIZATION - WEIGHT COMPARISON")
print("="*70)
print(f"\nL1 Accuracy: {acc_l1:.4f}")
print(f"L2 Accuracy: {acc_l2:.4f}")

print("\n" + "-"*70)
print("Weight Values Comparison:")
print("-"*70)
print(df.to_string(index=False))

# Count near-zero weights
l1_zeros = np.sum(np.abs(logreg_l1.coef_[0]) < 0.001)
l2_zeros = np.sum(np.abs(logreg_l2.coef_[0]) < 0.001)

print(f"\nL1 - Near-zero weights: {l1_zeros}/{len(feature_names)}")
print(f"L2 - Near-zero weights: {l2_zeros}/{len(feature_names)}")
print("\n" + "="*70)

# Create visualizations - Each in separate figure

# 1. Side-by-side weight comparison
fig1, ax1 = plt.subplots(figsize=(14, 6))
x = np.arange(len(feature_names))
width = 0.35
bars1 = ax1.bar(x - width/2, logreg_l1.coef_[0], width, label='L1', 
                color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
bars2 = ax1.bar(x + width/2, logreg_l2.coef_[0], width, label='L2', 
                color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
ax1.set_xlabel('Features', fontsize=12, fontweight='bold')
ax1.set_ylabel('Weight Value', fontsize=12, fontweight='bold')
ax1.set_title('L1 vs L2 Weights - All Features', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(feature_names, rotation=90, ha='right', fontsize=8)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# 2. Absolute weight comparison (scatter)
fig2, ax2 = plt.subplots(figsize=(10, 8))
abs_l1 = np.abs(logreg_l1.coef_[0])
abs_l2 = np.abs(logreg_l2.coef_[0])
scatter = ax2.scatter(abs_l1, abs_l2, s=150, alpha=0.6, c=range(len(feature_names)), 
                     cmap='viridis', edgecolors='black', linewidth=1.5)
max_val = max(abs_l1.max(), abs_l2.max())
ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2.5, label='y=x line', alpha=0.7)
ax2.set_xlabel('L1 Weight (Absolute)', fontsize=12, fontweight='bold')
ax2.set_ylabel('L2 Weight (Absolute)', fontsize=12, fontweight='bold')
ax2.set_title('L1 vs L2 Weight Magnitude', fontsize=14, fontweight='bold', pad=15)
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax2, label='Feature Index')
plt.tight_layout()
plt.show()

# 3. Weight distribution histogram
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.hist(logreg_l1.coef_[0], bins=20, alpha=0.7, label='L1', 
        color='#e74c3c', edgecolor='black', linewidth=1.2)
ax3.hist(logreg_l2.coef_[0], bins=20, alpha=0.7, label='L2', 
        color='#3498db', edgecolor='black', linewidth=1.2)
ax3.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax3.set_xlabel('Weight Value', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax3.set_title('Weight Distribution', fontsize=14, fontweight='bold', pad=15)
ax3.legend(fontsize=11)
ax3.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# 4. Top 10 features by L1 weight
fig4, ax4 = plt.subplots(figsize=(10, 8))
df_sorted = df.iloc[np.argsort(np.abs(logreg_l1.coef_[0]))[::-1][:10]]
y_pos = np.arange(len(df_sorted))
ax4.barh(y_pos - 0.2, df_sorted['L1 Weight'], 0.4, label='L1', 
        color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
ax4.barh(y_pos + 0.2, df_sorted['L2 Weight'], 0.4, label='L2', 
        color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
ax4.set_yticks(y_pos)
ax4.set_yticklabels(df_sorted['Feature'], fontsize=10)
ax4.set_xlabel('Weight Value', fontsize=12, fontweight='bold')
ax4.set_title('Top 10 Features (by L1 magnitude)', fontsize=14, fontweight='bold', pad=15)
ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax4.legend(fontsize=11)
ax4.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# 5. Heatmap of weights
fig5, ax5 = plt.subplots(figsize=(14, 4))
weight_matrix = np.column_stack([logreg_l1.coef_[0], logreg_l2.coef_[0]]).T
vmax = max(abs(logreg_l1.coef_[0].max()), abs(logreg_l2.coef_[0].max()))
im = ax5.imshow(weight_matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
ax5.set_yticks([0, 1])
ax5.set_yticklabels(['L1', 'L2'], fontsize=12, fontweight='bold')
ax5.set_xticks(range(len(feature_names)))
ax5.set_xticklabels(feature_names, rotation=90, ha='right', fontsize=8)
ax5.set_title('Weight Heatmap Comparison', fontsize=14, fontweight='bold', pad=15)
plt.colorbar(im, ax=ax5, label='Weight Value')
plt.tight_layout()
plt.show()

# 6. Sparsity comparison
fig6, ax6 = plt.subplots(figsize=(10, 6))
sparsity_data = {
    'L1 Regularization': [l1_zeros, len(feature_names) - l1_zeros],
    'L2 Regularization': [l2_zeros, len(feature_names) - l2_zeros]
}
x_labels = ['Near-Zero\nWeights', 'Non-Zero\nWeights']
x_pos = np.arange(len(x_labels))
width = 0.35
bars1 = ax6.bar(x_pos - width/2, sparsity_data['L1 Regularization'], width, 
               label='L1', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
bars2 = ax6.bar(x_pos + width/2, sparsity_data['L2 Regularization'], width, 
               label='L2', color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
ax6.set_xticks(x_pos)
ax6.set_xticklabels(x_labels, fontsize=11, fontweight='bold')
ax6.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
ax6.set_title('Sparsity Comparison (Weight Magnitude < 0.001)', fontsize=14, fontweight='bold', pad=15)
ax6.legend(fontsize=11)
ax6.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# Summary box plot
fig7, ax = plt.subplots(1, 1, figsize=(10, 6))
data_to_plot = [logreg_l1.coef_[0], logreg_l2.coef_[0]]
bp = ax.boxplot(data_to_plot, labels=['L1 Regularization', 'L2 Regularization'],
               patch_artist=True, widths=0.6,
               boxprops=dict(linewidth=2),
               medianprops=dict(color='red', linewidth=2.5),
               whiskerprops=dict(linewidth=2),
               capprops=dict(linewidth=2))

# Color the boxes
colors = ['#e74c3c', '#3498db']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Weight Value', fontsize=13, fontweight='bold')
ax.set_title('Weight Distribution Summary (Box Plot)', fontsize=15, fontweight='bold', pad=15)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print("\n✓ Visualizations completed!")