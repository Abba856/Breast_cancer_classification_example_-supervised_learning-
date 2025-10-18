# Breast Cancer Classification Example (Supervised Learning)

This repository is a small, self-contained educational example showing a typical supervised
learning workflow for binary classification using scikit-learn's breast cancer dataset.

The project trains multiple baseline classifiers, performs hyperparameter tuning for a
Random Forest, evaluates model performance with common metrics and plots, and saves the
best trained model (plus any preprocessing objects) to disk.

## Contents

- `breast_cancer_classification_example.ipynb` — Jupyter notebook with the full analysis and
	training pipeline. It contains data loading, exploratory summaries, preprocessing (scaling),
	training of baseline models (Logistic Regression, SVM, Random Forest), Random Forest tuning
	with GridSearchCV, evaluation (accuracy, precision, recall, F1, ROC AUC), and visualization
	(confusion matrix, ROC curve).
- `best_breast_cancer_model.joblib` — Serialized artifact (joblib) that stores the selected
	best model and associated preprocessing (for example, a fitted scaler). This allows quick
	inference without retraining.

## Quickstart

1. Create and activate a Python environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install the minimal dependencies:

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, the key packages are:

```bash
pip install scikit-learn pandas numpy matplotlib joblib
```

3. Open the notebook in Jupyter or VS Code and run the cells to reproduce the analysis:

```bash
jupyter notebook breast_cancer_classification_example.ipynb
```

Or convert the notebook to a script and run it (if you prefer a non-interactive run):

```bash
jupyter nbconvert --to script breast_cancer_classification_example.ipynb
python breast_cancer_classification_example.py
```
