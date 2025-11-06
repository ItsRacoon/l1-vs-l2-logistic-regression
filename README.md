# L1 vs L2 Regularization in Logistic Regression

## Project Overview

This project demonstrates the effect of **L1 (Lasso)** and **L2 (Ridge)** regularization on a logistic regression model using the **Breast Cancer Wisconsin dataset**. The goal is to compare how different penalties affect the **feature coefficients** and model performance.

## Dataset

- Source: `sklearn.datasets.load_breast_cancer`
- Features: 30 numeric features (tumor measurements)
- Target: Binary classification (malignant = 0, benign = 1)

## Steps Implemented

1. **Load dataset** using `scikit-learn`.
2. **Split dataset** into training and test sets.
3. **Standardize features** for regularization.
4. **Train logistic regression models**:

   * L1 regularization (`penalty='l1'`) → promotes sparsity
   * L2 regularization (`penalty='l2'`) → shrinks coefficients
5. **Compare coefficients** to observe sparsity and magnitude differences.
6. **Evaluate model accuracy** on the test set.
7. **Visualize coefficients** using matplotlib.

## Visualizations

Below are the generated plots that illustrate coefficient distributions, sparsity, and heatmaps comparing L1 and L2 regularization.

### Weight distribution

![Weight distribution](images/weight-distribution.png)

### Weight magnitude

![Weight magnitude](images/weight-magnitude.png)

### Weight heatmap

![Weight heatmap](images/weight-heatmap.png)

### Sparsity comparison (L1 vs L2)

![Sparsity comparison](images/sparsity-comparision.png)

### L1 — top 10 coefficients

# L1 vs L2 Regularization in Logistic Regression

## Project Overview

This repository compares L1 and L2 regularization applied to logistic regression. The code in `main.py` now uses the Wine dataset from scikit-learn and demonstrates differences in learned weights for a multiclass problem.

## Dataset (current code)

- Source: `sklearn.datasets.load_wine`
# L1 vs L2 Regularization in Logistic Regression

## Project Overview

This repository compares L1 and L2 regularization applied to logistic regression. The example code in `main.py` uses the Wine dataset from scikit-learn and demonstrates differences in learned weights for a multiclass problem.

## Dataset (current code)

- Source: `sklearn.datasets.load_wine`
- Features: 13 numeric features
- Target: Multiclass classification (3 wine classes)

## What changed

- The example was updated to use the Wine dataset (see `main.py`).
- Models are trained with `LogisticRegression` (solver='saga', `multi_class='ovr'`, `max_iter=5000`).
- The script prints class-specific weight comparisons and accuracy for L1 vs L2.

## Steps implemented by `main.py`

1. Load the Wine dataset.
2. Split into train/test sets (30% test).
3. Standardize features with `StandardScaler`.
4. Train logistic regression models with L1 and L2 penalties.
5. Print test accuracies and count near-zero weights.
6. Create two visualizations saved/shown by the script.

## Visualizations

Current image files in the `images/` folder are used by the README. Spaces in filenames are URL-encoded below so they render correctly on GitHub.

### Weight distribution (histogram)

![Weight distribution](images/weight%20distribution.png)

### Weight comparison (bar chart L1 vs L2)

![Weight comparison](images/weight%20comparison.png)

> If you add or rename files in `images/`, update these links or rename files to remove spaces (recommended).

## Usage

1. Clone the repo.
2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
# On Windows (PowerShell): venv\Scripts\Activate.ps1
# On Windows (cmd): venv\Scripts\activate.bat
source venv/bin/activate    # On UNIX-like systems
pip install -r requirements.txt
```

3. Run the script:

```bash
python main.py
```

The script prints L1 and L2 test accuracies and displays two plots (a histogram of weight distributions and a bar-chart comparison).

## Notes & next steps

- Consider renaming image files to remove spaces (e.g. `weight-distribution.png`) for portability.
- Optionally save the generated figures from `main.py` into `images/` with deterministic filenames.
- Add an example output block to the README after running the script once.

## Author

- Vishesh Kumar
