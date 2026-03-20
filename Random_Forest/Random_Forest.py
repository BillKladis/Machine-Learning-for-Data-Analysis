import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import category_encoders as ce

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report,
                              ConfusionMatrixDisplay, roc_auc_score,
                              RocCurveDisplay)

# ---------------- Data Loading ----------------
df = pd.read_csv("adult.csv", na_values=["?"], header=0, skiprows=[1])
df.dropna(inplace=True)

X = df.drop("income", axis=1)
Y = df["income"]

# ---------------- Split FIRST ----------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# ---------------- Encode AFTER split ----------------
cat_cols = ["workclass", "education", "marital-status", "occupation",
            "relationship", "race", "gender", "native-country"]
encoder = ce.BinaryEncoder(cols=cat_cols)
X_train = encoder.fit_transform(X_train)
X_test  = encoder.transform(X_test)

# NO scaling needed — Random Forest is tree-based, not distance-based

print("Preprocessing complete.")

# ---------------- Tune n_estimators with cross-validation ----------------
# Find the optimal number of trees before committing to a final model
n_tree_scores = {}
for n in range(50, 351, 50):
    rf_cv = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    scores = cross_val_score(
        rf_cv, X_train, Y_train,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='roc_auc'
    )
    n_tree_scores[n] = scores.mean()
    print(f"n_estimators={n:4d}  CV ROC-AUC={scores.mean():.4f}")

best_n = max(n_tree_scores, key=n_tree_scores.get)
print(f"\nBest n_estimators: {best_n},  CV ROC-AUC: {n_tree_scores[best_n]:.3f}")

# ---------------- Train final model ----------------
rf = RandomForestClassifier(
    class_weight='balanced',  # handle class imbalance by weighting classes inversely to frequency
    max_features='sqrt',  # common default
    n_estimators=best_n,
    max_depth=None,         # trees grow fully — RF handles overfitting via bagging
    min_samples_leaf=5,     # slightly smooths each tree, helps generalisation
    random_state=42,
    n_jobs=-1               # use all CPU cores
)
rf.fit(X_train, Y_train)
print("Model training complete.")

# ---------------- Confirm class order ----------------
pos_label = '>50K'
pos_index = list(rf.classes_).index(pos_label)
print(f"Class order: {rf.classes_},  pos_index={pos_index}")

# ---------------- Evaluate ----------------
Y_pred        = rf.predict(X_test)
Y_prob        = rf.predict_proba(X_test)[:, pos_index]
Y_test_binary = (Y_test == pos_label).astype(int)

print(f"\nTest ROC-AUC: {roc_auc_score(Y_test_binary, Y_prob):.3f}")
print(classification_report(Y_test, Y_pred))


# PLOT 1 — n_estimators vs ROC-AUC

fig, ax = plt.subplots(figsize=(8, 4))
ns   = list(n_tree_scores.keys())
aucs = list(n_tree_scores.values())
ax.plot(ns, aucs, marker='o', linewidth=2, color='steelblue', label='CV ROC-AUC')
ax.axvline(best_n, color='tomato', linestyle='--', linewidth=1.5,
           label=f'Best n = {best_n}')
ax.set_xlabel('Number of trees (n_estimators)')
ax.set_ylabel('Mean ROC-AUC  (5-fold CV)')
ax.set_title('Random Forest — choosing number of trees')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rf_plot_n_estimators.png', dpi=150)
plt.show()


# PLOT 2 — Confusion matrix

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    Y_test, Y_pred,
    display_labels=['<=50K', '>50K'],
    cmap='Blues',
    ax=ax
)
ax.set_title('Confusion matrix — Random Forest')
plt.tight_layout()
plt.savefig('rf_plot_confusion_matrix.png', dpi=150)
plt.show()


# PLOT 3 — ROC curve

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_predictions(
    Y_test, Y_prob,
    name=f'Random Forest  (n={best_n})',
    color='steelblue',
    pos_label=pos_label,
    ax=ax
)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random baseline')
ax.set_title('ROC curve — Random Forest')
ax.legend()
plt.tight_layout()
plt.savefig('rf_plot_roc_curve.png', dpi=150)
plt.show()


# PLOT 4 — Precision & Recall per class

report    = classification_report(Y_test, Y_pred, output_dict=True)
classes   = ['<=50K', '>50K']
precision = [report[c]['precision'] for c in classes]
recall    = [report[c]['recall']    for c in classes]

x     = np.arange(len(classes))
width = 0.35

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x - width/2, precision, width, label='Precision', color='steelblue')
ax.bar(x + width/2, recall,    width, label='Recall',    color='tomato')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Score')
ax.set_title('Precision and recall by class — Random Forest')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('rf_plot_precision_recall.png', dpi=150)
plt.show()


# PLOT 5 — Feature Importance (unique to Random Forest)

feature_names      = X_train.columns.tolist()
importances        = rf.feature_importances_
sorted_idx         = np.argsort(importances)[::-1][:15]   # top 15 features
sorted_importances = importances[sorted_idx]
sorted_names       = [feature_names[i] for i in sorted_idx]

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(sorted_names[::-1], sorted_importances[::-1], color='steelblue')
ax.set_xlabel('Importance score (mean decrease in impurity)')
ax.set_title('Top 15 most important features — Random Forest')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('rf_plot_feature_importance.png', dpi=150)
plt.show()