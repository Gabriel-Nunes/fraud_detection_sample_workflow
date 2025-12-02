# Fraud Detection Models Comparison Notebook

This notebook trains and compares **Logistic Regression**, **Random Forest**, **XGBoost** and **LightGBM** on a PCA-transformed dataset for fraud detection. It contains:

- Clear markdown sections so it can be exported to an `.ipynb` file.
- Line-by-line explanations and comments for almost every code block.
- Plots: ROC and Precision-Recall curves, confusion matrices.
- A results summary table.

---

## 1. Introduction

This notebook assumes you already have a pandas DataFrame `df` with the following columns:
- `pca_1` ... `pca_20` (PCA components)
- `value` (transaction amount)
- `is_fraud` (binary target, 0/1)

We will:
1. Split the data (stratified) into train/test.
2. Train four models using reasonable default hyperparameters.
3. Evaluate with classification report, ROC AUC and PR AUC.
4. Show comparative plots and a summary table.

---

```python
# 0. Required imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

# Note: this notebook uses only commonly available libraries. Make sure xgboost and lightgbm are installed.
```

---

## 2. Load data (placeholder)

**Explanation:** Here you should load your dataset. The example keeps a placeholder comment so notebook is reusable.

```python
# Replace this with the actual loading step for your environment
# Example: df = pd.read_csv('pca_transactions.csv')
# For the purposes of this notebook we assume `df` is already in the workspace

# Quick check (uncomment when df is available):
# display(df.sample(5))
# df.info()
# df.describe()
```

**Line-by-line explanation:**
- `pd.read_csv(...)`: reads CSV into a DataFrame.
- `display(df.sample(5))`: shows 5 random rows for sanity check.
- `df.info()`: shows dtypes and null counts.
- `df.describe()`: shows descriptive statistics.

---

## 3. Prepare features and target

**Rationale:** We will keep all PCA components plus `value` as features. PCA output is already centered/scaled by construction — do not rescale the PCA features again.

```python
# 3.1 Define X and y
X = df.drop(columns=['is_fraud'])  # features dataframe
y = df['is_fraud']                 # target series

# 3.2 Train-test split (stratified to keep class ratio)
# Explanation: stratify=y ensures the same fraud ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Print sizes
print('Train shape:', X_train.shape, 'Test shape:', X_test.shape)
print('Train fraud ratio:', y_train.mean(), 'Test fraud ratio:', y_test.mean())
```

**Line-by-line explanation:**
- `drop(columns=['is_fraud'])`: builds the features matrix.
- `train_test_split(...)`: splits dataset into train and test. `test_size=0.25` means 25% for test.
- `stratify=y`: preserves class distribution in both splits.
- `random_state=42`: reproducible split.

---

## 4. Utility functions (metrics & plots)

```python
# 4.1 Generic evaluation function

def eval_model_print(model, X_test, y_test):
    """Print classification report and numeric metrics.
    model: trained estimator with predict and predict_proba
    """
    # Predict hard labels
    y_pred = model.predict(X_test)
    # Predict probabilities for positive class
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # some models (rare) only provide decision_function; handle that
        try:
            y_score = model.decision_function(X_test)
            # convert to 0..1 with a sigmoid (approx) only for metrics that need a score
            y_prob = 1 / (1 + np.exp(-y_score))
        except Exception:
            y_prob = y_pred  # fallback (not ideal for ROC)

    # Print classification report (precision, recall, f1)
    print(classification_report(y_test, y_pred, digits=4))
    # Numeric metrics
    try:
        roc = roc_auc_score(y_test, y_prob)
    except Exception:
        roc = np.nan
    try:
        pr = average_precision_score(y_test, y_prob)
    except Exception:
        pr = np.nan
    print(f"ROC AUC: {roc:.4f}")
    print(f"PR AUC:  {pr:.4f}")


# 4.2 Plotting helper: ROC and PR for multiple models

def plot_roc_pr(models, X_test, y_test, figsize=(12,5)):
    """Plot ROC and Precision-Recall for a dict of fitted models.
    models: dict{name: model}
    """
    plt.figure(figsize=figsize)
    # ROC subplot
    plt.subplot(1,2,1)
    for name, model in models.items():
        try:
            RocCurveDisplay.from_estimator(model, X_test, y_test, name=name)
        except Exception as e:
            print(f"Could not plot ROC for {name}: {e}")
    plt.title('ROC Curves')

    # PR subplot
    plt.subplot(1,2,2)
    for name, model in models.items():
        try:
            PrecisionRecallDisplay.from_estimator(model, X_test, y_test, name=name)
        except Exception as e:
            print(f"Could not plot PR for {name}: {e}")
    plt.title('Precision-Recall Curves')
    plt.tight_layout()
    plt.show()


# 4.3 Confusion matrix plot

def plot_confusion_matrix(model, X_test, y_test, normalize=False):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, normalize='true' if normalize else None)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix' + (' (normalized)' if normalize else ''))
    plt.colorbar()
    ticks = [0,1]
    plt.xticks(ticks, ['non-fraud','fraud'])
    plt.yticks(ticks, ['non-fraud','fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i,j]
            if normalize:
                txt = f"{val:.2f}"
            else:
                txt = f"{int(val)}"
            plt.text(j, i, txt, ha='center', va='center', color='white' if cm.max()>0.5 else 'black')
    plt.show()
```

**Line-by-line explanation:**
- `eval_model_print`: prints classification report and computes ROC and PR AUC from predicted probabilities.
- `plot_roc_pr`: draws both ROC and PR curves side-by-side for comparison.
- `plot_confusion_matrix`: simple visualization of confusion matrix; `normalize=True` shows rates.

---

## 5. Train models

We will train four models with sensible defaults. For the boosting models we will set `scale_pos_weight` (XGBoost) and `class_weight` or `is_unbalance` equivalents for LightGBM when appropriate.

```python
# 5.1 Compute class imbalance ratio to help with model hyperparameters
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
print('Negatives (train):', neg)
print('Positives (train):', pos)
scale_pos_weight = neg / pos
print('scale_pos_weight (neg/pos):', scale_pos_weight)
```

**Explanation:**
- `scale_pos_weight` is used by XGBoost to inform the objective about the imbalance. It's the ratio of negative to positive samples.


### 5.2 Logistic Regression (baseline)

```python
logreg = LogisticRegression(
    class_weight='balanced',  # penalize the majority class automatically
    max_iter=500,
    n_jobs=-1,
    random_state=42
)
logreg.fit(X_train, y_train)
print('Logistic Regression trained')

eval_model_print(logreg, X_test, y_test)
plot_confusion_matrix(logreg, X_test, y_test, normalize=True)
```

**Line-by-line notes:**
- `class_weight='balanced'`: increases the weight of minority class (fraud) during training.
- `max_iter=500`: ensure convergence for some datasets.


### 5.3 Random Forest

```python
rf = RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced_subsample',  # balance per tree
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
print('Random Forest trained')

eval_model_print(rf, X_test, y_test)
plot_confusion_matrix(rf, X_test, y_test, normalize=True)
```

**Notes:**
- `balanced_subsample` uses the class distribution of the bootstrap sample for each tree — often helpful.
- RFs are robust and provide good recall in many fraud tasks.


### 5.4 XGBoost

```python
xgb = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,  # handle imbalance
    n_jobs=-1,
    random_state=42,
    use_label_encoder=False
)

xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print('XGBoost trained')

eval_model_print(xgb, X_test, y_test)
plot_confusion_matrix(xgb, X_test, y_test, normalize=True)
```

**Line-by-line notes:**
- `use_label_encoder=False`: avoids deprecated label encoder in recent xgboost versions.
- `scale_pos_weight`: critical for imbalanced classification with XGBoost.


### 5.5 LightGBM

We train LightGBM with `is_unbalance=True` or `scale_pos_weight` depending on API. Using `is_unbalance=True` makes LightGBM internally treat the classes as unbalanced. We can also provide `class_weight` or `scale_pos_weight` if desired.

```python
# LightGBM classifier using sklearn API
lgbm = lgb.
(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=-1,
    subsample=0.9,

    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    is_unbalance=True  # quick handling of imbalance
)

# Fit
lgbm.fit(X_train, y_train)
print('LightGBM trained')

eval_model_print(lgbm, X_test, y_test)
plot_confusion_matrix(lgbm, X_test, y_test, normalize=True)
```

**Notes:**
- `is_unbalance=True` tells LightGBM to use the inverse class frequency as weights.
- LightGBM is often faster than XGBoost and performs similarly on tabular data.

---

## 6. Comparative plots and summary

```python
models = {
    'Logistic Regression': logreg,
    'Random Forest': rf,
    'XGBoost': xgb,
    'LightGBM': lgbm
}

# Summary table
results = []
for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        try:
            y_score = model.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-y_score))
        except Exception:
            y_prob = model.predict(X_test)
    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)
    results.append({'Model': name, 'ROC_AUC': roc, 'PR_AUC': pr})

summary_df = pd.DataFrame(results).sort_values('PR_AUC', ascending=False).reset_index(drop=True)
print(summary_df)

# Plot ROC and PR
plot_roc_pr(models, X_test, y_test)
```

**Explanation:**
- We rank models by PR_AUC (average precision) because Precision-Recall is more informative on imbalanced datasets.

---

## 7. Next steps / suggestions

1. **Hyperparameter tuning:** use `RandomizedSearchCV` or `Optuna` for each model (especially XGBoost and LightGBM). Use stratified CV.
2. **Threshold optimization:** choose decision threshold to maximize business metric (e.g., maximize recall at acceptable precision). Use `precision_recall_curve` to find thresholds.
3. **Feature importance:** although PCA components are linear combinations and less interpretable, you can analyze which components contribute more using tree-based feature importance.
4. **Calibration:** for probabilistic outputs, check calibration (Platt scaling or isotonic regression) if you need well-calibrated probabilities.
5. **Advanced sampling:** try SMOTE, ADASYN or ensemble approaches (balanced bagging) if you want to synthetically augment minority class — but be careful applying these to PCA components.

---

## 8. Save models (optional)

```python
# Example: save best model with joblib
import joblib

best_model_name = summary_df.loc[0, 'Model']
best_model = models[best_model_name]
joblib.dump(best_model, 'best_fraud_model.joblib')
print('Saved', best_model_name)
```

---

## Appendix: Short explanations (quick reference)

- **Precision:** Of predicted frauds, how many were actual frauds (TP / (TP + FP)).
- **Recall (Sensitivity):** Of real frauds, how many did we detect (TP / (TP + FN)).
- **F1-score:** Harmonic mean between precision and recall.
- **ROC AUC:** Measures ranking quality (how well positives are ordered above negatives). Can be misleading on imbalanced data.
- **PR AUC (Average Precision):** Better signal for imbalanced datasets: focuses on performance for the positive class.

---

# End of notebook

# You can now export this content to an .ipynb file or continue editing.
