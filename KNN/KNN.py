import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import category_encoders as ce

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report,
                              ConfusionMatrixDisplay, roc_auc_score,
                              RocCurveDisplay)

# ---------------- Data Loading ----------------
df = pd.read_csv("adult.csv", na_values=["?"], header=0, skiprows=[1])
df.dropna(inplace=True)

X = df.drop("income", axis=1)
Y = df["income"]

# ---------------- Split FIRST — nothing done to data yet ----------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# ---------------- Encode AFTER split ----------------
cat_cols = ["workclass", "education", "marital-status", "occupation",
            "relationship", "race", "gender", "native-country"]
encoder = ce.BinaryEncoder(cols=cat_cols)
X_train = encoder.fit_transform(X_train)  # learns mapping from train only
X_test  = encoder.transform(X_test)       # applies same mapping, no learning

# ---------------- Scale ----------------
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("Preprocessing complete.")

# ---------------- Tune k ----------------
k_scores = {}
for k in range(1, 31, 2):
    knn_cv = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(
        knn_cv, X_train_scaled, Y_train,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='roc_auc'
    )
    k_scores[k] = scores.mean()

best_k = max(k_scores, key=k_scores.get)
print(f"Best k: {best_k},  CV ROC-AUC: {k_scores[best_k]:.3f}")

# ---------------- Train final model ----------------
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, Y_train)

# ---------------- Confirm class order ----------------
# This tells us which index corresponds to '>50K' so Y_prob is correct
pos_label  = '>50K'
pos_index  = list(knn.classes_).index(pos_label)
print(f"Class order: {knn.classes_},  pos_index={pos_index}")

# ---------------- Evaluate ----------------
Y_pred = knn.predict(X_test_scaled)
Y_prob = knn.predict_proba(X_test_scaled)[:, pos_index]

print(f"Test ROC-AUC: {roc_auc_score(Y_test, Y_prob):.3f}")
print(classification_report(Y_test, Y_pred))


# PLOT 1 — k vs ROC-AUC (finding the best k)

fig, ax = plt.subplots(figsize=(8, 4))
ks   = list(k_scores.keys())
aucs = list(k_scores.values())
ax.plot(ks, aucs, marker='o', linewidth=2, color='steelblue', label='CV ROC-AUC')
ax.axvline(best_k, color='tomato', linestyle='--', linewidth=1.5,
           label=f'Best k = {best_k}')
ax.set_xlabel('k  (number of neighbours)')
ax.set_ylabel('Mean ROC-AUC  (5-fold CV)')
ax.set_title('KNN — choosing the best k')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot_k_tuning.png', dpi=150)
plt.show()

# PLOT 2 — Confusion matrix
#-------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    Y_test, Y_pred,
    display_labels=['<=50K', '>50K'],
    cmap='Blues',
    ax=ax
)
ax.set_title('Confusion matrix')
plt.tight_layout()
plt.savefig('plot_confusion_matrix.png', dpi=150)
plt.show()


# PLOT 3 — ROC curve  (pos_label fixes the error)
#-------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_predictions(
    Y_test, Y_prob,
    name=f'KNN  (k={best_k})',
    color='steelblue',
    pos_label=pos_label,          # ← fixes the ValueError you got
    ax=ax
)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random baseline')
ax.set_title('ROC curve')
ax.legend()
plt.tight_layout()
plt.savefig('plot_roc_curve.png', dpi=150)
plt.show()

# ================================================
# PLOT 4 — Precision & Recall per class
# ================================================
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
ax.set_title('Precision and recall by class')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plot_precision_recall.png', dpi=150)
plt.show()