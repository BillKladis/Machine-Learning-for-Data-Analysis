import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import category_encoders as ce

from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Data loading
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# Count languages and frameworks instead of encoding raw comma-separated strings
def count_items(series):
    return series.fillna("").apply(lambda x: len([i for i in x.split(",") if i.strip()]))

for df in [train_df, test_df]:
    df["n_languages"]  = count_items(df["languages"])
    df["n_frameworks"] = count_items(df["frameworks"])

train_df.drop(columns=["languages", "frameworks"], inplace=True)
test_df.drop( columns=["languages", "frameworks"], inplace=True)

# Separate features and target
TARGET = "salary_usd"

X_train = train_df.drop(TARGET, axis=1)
Y_train = train_df[TARGET]
X_test  = test_df.drop(TARGET, axis=1)
Y_test  = test_df[TARGET]

# Encode categorical columns — fit only on train to prevent leakage
cat_cols = ["country", "education", "company_size"]

encoder = ce.BinaryEncoder(cols=cat_cols)
X_train = encoder.fit_transform(X_train)
X_test  = encoder.transform(X_test)

# Scale for linear and lasso — random forest does not need scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Linear regression
lr = LinearRegression()
lr.fit(X_train_scaled, Y_train)

Y_pred_lr = lr.predict(X_test_scaled)
lr_r2     = r2_score(Y_test, Y_pred_lr)
lr_rmse   = np.sqrt(mean_squared_error(Y_test, Y_pred_lr))

print("Linear Regression")
print(f"  R2:   {lr_r2:.3f}")
print(f"  RMSE: {lr_rmse:,.0f}")

# Lasso — LassoCV finds the best lambda automatically via cross-validation
lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso.fit(X_train_scaled, Y_train)

Y_pred_lasso = lasso.predict(X_test_scaled)
lasso_r2     = r2_score(Y_test, Y_pred_lasso)
lasso_rmse   = np.sqrt(mean_squared_error(Y_test, Y_pred_lasso))
n_used       = np.sum(lasso.coef_ != 0)
n_total      = X_train.shape[1]

print("Lasso")
print(f"  R2:             {lasso_r2:.3f}")
print(f"  RMSE:           {lasso_rmse:,.0f}")
print(f"  Best lambda:    {lasso.alpha_:.4f}")
print(f"  Features kept:  {n_used} of {n_total}")

# Random forest — unscaled data, trees are not distance-based
rf = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, Y_train)

Y_pred_rf = rf.predict(X_test)
rf_r2     = r2_score(Y_test, Y_pred_rf)
rf_rmse   = np.sqrt(mean_squared_error(Y_test, Y_pred_rf))

print("Random Forest")
print(f"  R2:   {rf_r2:.3f}")
print(f"  RMSE: {rf_rmse:,.0f}")

# Collect results
models      = ["Linear Regression", "Lasso", "Random Forest"]
r2_scores   = [lr_r2,   lasso_r2,   rf_r2]
rmse_scores = [lr_rmse, lasso_rmse, rf_rmse]
preds       = [Y_pred_lr, Y_pred_lasso, Y_pred_rf]
colors      = ["steelblue", "tomato", "seagreen"]

# Plot 1 — R2 comparison across models
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(models, r2_scores, color=colors, width=0.5)
ax.set_ylabel("R2 (higher is better, 1.0 = perfect)")
ax.set_title("Model comparison - R2")
ax.set_ylim(min(0, min(r2_scores)) - 0.05, 1.05)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, r2_scores):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.savefig("plot_r2_comparison.png", dpi=150)
plt.show()

# Plot 2 — RMSE comparison across models
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(models, rmse_scores, color=colors, width=0.5)
ax.set_ylabel("RMSE (lower is better, in USD)")
ax.set_title("Model comparison - RMSE")
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, rmse_scores):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 500,
            f"{val:,.0f}",
            ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig("plot_rmse_comparison.png", dpi=150)
plt.show()

# Plot 3 — Predicted vs actual salary for each model
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, name, pred, color in zip(axes, models, preds, colors):
    ax.scatter(Y_test, pred, alpha=0.5, color=color, edgecolors='none', s=20)
    min_val = min(Y_test.min(), pred.min())
    max_val = max(Y_test.max(), pred.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            'k--', linewidth=1, label='Perfect prediction')
    ax.set_xlabel("Actual salary")
    ax.set_ylabel("Predicted salary")
    ax.set_title(f"{name}  R2={r2_score(Y_test, pred):.3f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

plt.suptitle("Predicted vs actual salary", y=1.02)
plt.tight_layout()
plt.savefig("plot_predicted_vs_actual.png", dpi=150)
plt.show()

# Plot 4 — Residuals for each model
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, name, pred, color in zip(axes, models, preds, colors):
    residuals = Y_test - pred
    ax.scatter(pred, residuals, alpha=0.5, color=color, edgecolors='none', s=20)
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    ax.set_xlabel("Predicted salary")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title(f"{name} - residuals")
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("plot_residuals.png", dpi=150)
plt.show()

# Plot 5 — Lasso non-zero coefficients
feature_names = X_train.columns.tolist()
coef_df = pd.DataFrame({
    "feature":     feature_names,
    "coefficient": lasso.coef_
}).query("coefficient != 0").sort_values("coefficient")

fig, ax = plt.subplots(figsize=(7, max(4, len(coef_df) * 0.4)))
coef_colors = ["tomato" if c < 0 else "steelblue" for c in coef_df["coefficient"]]
ax.barh(coef_df["feature"], coef_df["coefficient"], color=coef_colors)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel("Coefficient value")
ax.set_title(f"Lasso coefficients  ({len(coef_df)} of {n_total} features kept)")
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("plot_lasso_coefficients.png", dpi=150)
plt.show()

# Plot 6 — Random forest feature importance
importances   = rf.feature_importances_
sorted_idx    = np.argsort(importances)[::-1][:15]
sorted_names  = [feature_names[i] for i in sorted_idx]
sorted_values = importances[sorted_idx]

fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(sorted_names[::-1], sorted_values[::-1], color="seagreen")
ax.set_xlabel("Importance score")
ax.set_title("Random Forest - top 15 features")
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("plot_rf_importance.png", dpi=150)
plt.show()