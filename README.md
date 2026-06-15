# Machine Learning for Data Analysis

A structured collection of supervised-learning and feature-analysis projects built toward professional standards. Each module demonstrates a complete pipeline: raw data → preprocessing → model training → hyperparameter tuning → evaluation → visualisation. The emphasis throughout is on **correctness** (no data leakage, proper splits), **rigor** (cross-validated metrics, multiple evaluation angles), and **reasoning** (every design decision is explained).

---

## Table of Contents

1. [Repository Layout](#repository-layout)
2. [Cross-Cutting Design Principles](#cross-cutting-design-principles)
3. [Project: KNN Classification (Adult Income)](#1-knn-classification--adult-income)
4. [Project: Naive Bayes (Diabetes)](#2-naive-bayes--diabetes-classification)
5. [Project: Random Forest Classification (Adult Income)](#3-random-forest-classification--adult-income)
6. [Project: Regression Comparison (Developer Salary)](#4-regression-comparison--developer-salary)
7. [Project: Mutual Information – Simple](#5-mutual-information--simple-feature-ranking)
8. [Project: Bank Churn – Demographic & Feature Analysis](#6-bank-churn--demographic--feature-analysis)
9. [Datasets at a Glance](#datasets-at-a-glance)
10. [Mathematical Reference](#mathematical-reference)
11. [System Design Overview](#system-design-overview)

---

## Repository Layout

```
Machine-Learning-for-Data-Analysis/
│
├── ames.csv                              # Ames Housing dataset (81 cols, ~1 460 rows)
│
├── KNN/
│   ├── KNN.py                            # Full KNN pipeline
│   ├── adult.csv                         # UCI Adult Census Income dataset
│   ├── README.md
│   └── plot_*.png                        # k-tuning, ROC, confusion, PR curves
│
├── Bayes/
│   ├── Naive-Bayes.py                    # Gaussian NB pipeline
│   ├── Naive-Bayes-Classification-Data.csv
│   └── diabetes.pbix                     # Power BI dashboard
│
├── Random_Forest/
│   ├── Random_Forest.py                  # RF pipeline + feature importance
│   ├── adult.csv
│   ├── README.md
│   └── rf_plot_*.png                     # n-estimators, ROC, confusion, PR, importance
│
├── Lasso vs Linear vs Random Tree/
│   ├── Comparison of common regressions.py
│   ├── train.csv / test.csv              # Developer salary dataset
│   ├── data_dictionary.csv
│   ├── README.md
│   └── *.png                             # Residuals, coefficients, feature importance
│
├── Mutual_Information/
│   ├── Mutual_info.py                    # MI-based feature ranking
│   ├── Naive-Bayes-Classification-Data.csv
│   └── adult.csv
│
└── Bank_Churning_Demographic_Analysis/
    ├── Data_editting_for visual.py       # Preprocessing for Power BI
    ├── KMeans_feature_eng.py             # K-Means customer segmentation
    ├── Mutual_Information.py             # Advanced MI analysis
    ├── Bank_Churn.csv
    ├── Bank_Churn_edited.csv
    └── PowerBI.csv
```

---

## Cross-Cutting Design Principles

These principles are applied consistently across every project.

### 1. Leakage-Free Preprocessing

All transformers (scalers, encoders) are **fit on the training set only** and then applied to both train and test. Fitting on the full dataset before splitting causes data leakage: the model indirectly "sees" the test labels through the fitted transformer's statistics, inflating reported performance.

```
raw data
   │
   ├── train split ──► fit encoder / scaler on TRAIN ──► transform TRAIN
   │                                                  └──► transform TEST
   └── test split  ──────────────────────────────────────────────────────►
```

### 2. Stratified Splitting for Class Imbalance

`train_test_split(..., stratify=y)` preserves the class-proportion of the original dataset in every split. Without stratification, a random split can produce a test set with a very different ratio of positives to negatives, making evaluation unstable.

### 3. Cross-Validation for Hyperparameter Selection

Hyperparameters (k in KNN, n_estimators in RF, λ in Lasso) are selected with **StratifiedKFold cross-validation** on the training set. The test set is touched **exactly once** for final evaluation, ensuring the reported metric is unbiased.

### 4. ROC-AUC as the Primary Classification Metric

Accuracy is misleading on imbalanced datasets. ROC-AUC summarises the trade-off between the true-positive rate and false-positive rate across every possible threshold, making it robust to class imbalance.

### 5. Categorical Encoding Strategy

`BinaryEncoder` (from `category_encoders`) is used instead of one-hot encoding for high-cardinality columns. Binary encoding converts a category with C unique values into ⌈log₂C⌉ binary columns instead of C columns, reducing dimensionality while preserving category distinctness.

---

## 1. KNN Classification — Adult Income

**Location**: `KNN/KNN.py`

### Task

Binary classification: predict income >$50K from census attributes (UCI Adult, ~30K rows, 15 features). Target: income (<=50K or >50K).

### Preprocessing Pipeline

```
Categorical columns (8)  ──► BinaryEncoder (fit on train) ──► binary feature matrix
Numeric columns          ──► MinMaxScaler  (fit on train) ──► scaled to [0, 1]
                                                                      │
                                                                 concatenated
                                                                      │
                                                              KNeighborsClassifier
```

MinMaxScaler is essential for KNN because the algorithm computes Euclidean distance. Without scaling, high-magnitude features (e.g. capital-gain up to 99 999) dominate the distance, effectively ignoring low-magnitude features.

**MinMax formula**:

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

### Mathematical Foundation — KNN

KNN is a non-parametric, instance-based learner. For a query point **x**, it:

1. Computes the Euclidean distance to every training point:

$$d(\mathbf{x}, \mathbf{x}_i) = \sqrt{\sum_{j=1}^{p}(x_j - x_{ij})^2}$$

2. Identifies the k nearest neighbours N_k(**x**).
3. Assigns the majority class:

$$\hat{y} = \arg\max_{c} \sum_{i \in N_k(\mathbf{x})} \mathbf{1}[y_i = c]$$

**No training phase** — inference cost is O(n · p) per query.

### Hyperparameter Tuning

Swept k ∈ [1, 31] (odd values prevent ties) via 5-fold stratified CV on training ROC-AUC. **Result**: k = 29, ROC-AUC ≈ 0.88. Small k overfits; large k over-smooths; k=29 balances both.

### Results

ROC-AUC: ~0.88. **Outputs**: confusion matrix, ROC curve, precision-recall plots.

---

## 2. Naive Bayes — Diabetes Classification

**Location**: `Bayes/Naive-Bayes.py`

### Task

Binary classification: predict diabetes (0/1) from glucose and blood pressure (`Naive-Bayes-Classification-Data.csv`).

### Mathematical Foundation — Gaussian Naive Bayes

Naive Bayes applies Bayes' theorem with the **conditional independence** assumption among features given the class:

$$P(y \mid \mathbf{x}) \propto P(y) \prod_{j=1}^{p} P(x_j \mid y)$$

For continuous features the likelihood is modelled as a Gaussian:

$$P(x_j \mid y = c) = \frac{1}{\sqrt{2\pi\sigma_{cj}^2}} \exp\!\left(-\frac{(x_j - \mu_{cj})^2}{2\sigma_{cj}^2}\right)$$

where μ_cj and σ²_cj are estimated as the sample mean and variance of feature j within class c on the training data.

**Prediction**:

$$\hat{y} = \arg\max_{c}\left[\log P(y=c) + \sum_{j=1}^{p} \log P(x_j \mid y=c)\right]$$

Working in log-space avoids numerical underflow when multiplying many small probabilities.

### Feature Importance

Computed via mean class-conditional difference: $\text{FI}_j = |\mu_{1j} - \mu_{0j}|$. Measures how much each feature separates positive vs. negative classes.

### Design Decisions

- **Stratified 80/20 split**: preserves the diabetes prevalence ratio in both sets.
- **GaussianNB over MultinomialNB**: the features are continuous real values, not counts.
- **No scaling**: Naive Bayes estimates per-class Gaussians directly from the raw values — scaling would be redundant.

---

## 3. Random Forest Classification — Adult Income

**Location**: `Random_Forest/Random_Forest.py`

### Task

Same binary income classification as KNN, enabling direct model comparison on identical data.

### Mathematical Foundation — Random Forest

Random Forest is an **ensemble of decision trees** trained with two sources of randomness:

**1. Bootstrap aggregation (Bagging)**

Each tree is trained on a bootstrap sample (n rows sampled with replacement from n training rows). Because ~37% of rows are excluded from each bootstrap sample (out-of-bag rows), variance is reduced without additional data.

**2. Random feature subsets (Feature Randomness)**

At each split, only `max_features = sqrt(p)` features are considered (≈ 7 out of 108 after encoding). This decorrelates the trees, reducing variance further.

**Prediction** (majority vote):

$$\hat{y} = \text{mode}\{h_t(\mathbf{x}) : t = 1, \ldots, T\}$$

**Feature Importance — Mean Decrease in Impurity (MDI)**:

$$\text{FI}(j) = \frac{1}{T}\sum_{t=1}^{T}\sum_{s \in \text{splits on } j} \frac{n_s}{n}\cdot\Delta\text{Gini}(s)$$

where Δ Gini(s) is the weighted reduction in Gini impurity at split s, n_s is the number of samples at that node, and n is total training samples.

**Gini Impurity**:

$$G = 1 - \sum_{c} p_c^2$$

### Hyperparameter Roles

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_estimators | tuned via CV (50–350) | More trees → lower variance; diminishing returns after ~200 |
| max_features | 'sqrt' | Standard heuristic; reduces tree correlation |
| max_depth | None | Full trees; bagging controls variance instead of pruning |
| min_samples_leaf | 5 | Slight regularisation against very small leaf nodes |
| class_weight | 'balanced' | Upweights minority class (>50K) automatically |

**class_weight='balanced'** scales sample weights by n / (n_classes × count_c), compensating for the ~75/25 class imbalance in the Adult dataset.

### Result

ROC-AUC: ~0.91 (vs. KNN's 0.88). RF outperforms because it builds non-linear, axis-aligned boundaries without distance-based scale sensitivity.

---

## 4. Regression Comparison — Developer Salary

**Location**: `Lasso vs Linear vs Random Tree/Comparison of common regressions.py`

### Task

Regression: predict a software engineer's annual salary (USD) from experience, country, education, languages, frameworks, and company size.

**Dataset** (`train.csv` / `test.csv`):

| Feature | Type | Engineering |
|---------|------|-------------|
| experience | Numeric (years) | Used directly |
| country | Categorical | BinaryEncoder |
| education | Categorical | BinaryEncoder |
| languages | String (comma-sep list) | → `n_languages` (count) |
| frameworks | String (comma-sep list) | → `n_frameworks` (count) |
| company_size | Categorical | BinaryEncoder |
| salary_usd | Numeric — **target** | — |

### Feature Engineering

The `count_items()` helper converts comma-separated strings like `"Python,JavaScript,Rust"` into the integer count `3`. This extracts signal (breadth of skills) while avoiding a high-cardinality multi-label explosion.

### Models

#### Linear Regression

Minimises MSE: $\mathcal{L}(\boldsymbol{\beta}) = \frac{1}{n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2$. Requires MinMaxScaler for solver convergence.

#### Lasso Regression

Adds L1 penalty for automatic feature selection: $\mathcal{L}(\boldsymbol{\beta}) = \frac{1}{n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda\|\boldsymbol{\beta}\|_1$. The L1 corner at β_j = 0 induces sparsity; weak features get zeroed. λ tuned via LassoCV.

#### Random Forest Regressor

Ensemble of trees predicting the mean of leaf values: $\hat{y} = \frac{1}{T}\sum_{t=1}^{T} h_t(\mathbf{x})$. No scaling needed.

### Evaluation Metrics

**R²**: $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ (fraction of variance explained, 0–1 scale).

**RMSE**: $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$ (same units as target).

**Residual plots**: Reveal heteroscedasticity, non-linearity, and outliers.

---

## 5. Mutual Information — Simple Feature Ranking

**Location**: `Mutual_Information/Mutual_info.py`

### Task

Rank features by their statistical dependence with the target, prior to or independently of any model.

**Dataset**: same as Naive Bayes (`Naive-Bayes-Classification-Data.csv`).

### Mathematical Foundation — Mutual Information

Mutual Information (MI) quantifies how much knowing a feature X reduces uncertainty about the target Y:

$$I(X; Y) = \sum_{x}\sum_{y} P(x, y)\log\frac{P(x, y)}{P(x)P(y)}$$

For continuous X the sum becomes an integral. Sklearn's `mutual_info_classif` estimates MI using k-nearest-neighbour density estimation (Kraskov estimator), which is non-parametric and can detect **non-linear** dependencies that correlation (Pearson r) would miss.

**Key properties**:
- I(X; Y) ≥ 0 always
- I(X; Y) = 0 iff X and Y are independent
- No assumption of linearity or Gaussianity

### Implementation

Uses k-NN density estimation (Kraskov estimator). Properly classify features as discrete vs. continuous to avoid MI inflation. Complements MI ranking with boxplots, scatter plots, and correlation heatmaps.

---

## 6. Bank Churn — Demographic & Feature Analysis

**Location**: `Bank_Churning_Demographic_Analysis/`

### Task

Understand which customer attributes drive churn and engineer features for downstream models.

**Dataset** (`Bank_Churn.csv`, ~10 000 rows):

| Feature | Type | Description |
|---------|------|-------------|
| CreditScore | Numeric | Customer credit score |
| Geography | Categorical | Country (France, Germany, Spain) |
| Gender | Categorical | Male / Female |
| Age | Numeric | Customer age |
| Tenure | Numeric | Years as a customer |
| Balance | Numeric | Account balance (zero-heavy) |
| NumOfProducts | Integer | Number of bank products held |
| HasCrCard | Binary | Has credit card |
| IsActiveMember | Binary | Active in last period |
| EstimatedSalary | Numeric | Annual salary estimate |
| Exited | Binary target | 1 = churned |

### Script 1 — Data Preprocessing

Prepares dataset for Power BI: boolean `Exited_Bool`, pre-aggregated churn rates by age/gender, float32 type casting for memory efficiency. Output: `Bank_Churn_edited.csv` → `PowerBI.csv`.

### Script 2 — K-Means Customer Segmentation (`KMeans_feature_eng.py`)

**Features used**: Balance, EstimatedSalary, CreditScore (scaled with MinMaxScaler).

**K-Means objective** — minimise within-cluster sum of squared distances:

$$J = \sum_{k=1}^{K}\sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|_2^2$$

**Algorithm**: Lloyd's algorithm with k-means++ initialisation. Parameters: `n_clusters=6`, `n_init=50` (50 random restarts for robustness). Profiles each cluster by mean values and churn rate to identify customer personas.

**Outputs**: cluster profile heatmap + churn-rate-per-cluster bar chart.

### Script 3 — Advanced Mutual Information (`Mutual_Information.py`)

A four-stage analytical pipeline:

```
Load & Clean
     │
     ├── drop: CustomerId, Surname (IDs, not signals)
     ├── factorize: Geography, Gender (string → int)
     └── drop pre-aggregated rate columns (would leak target info)
     │
Display Distributions
     │  (histogram + KDE for every feature — detect skew, zero-inflation)
     │
MI Scores (two passes)
     │  1. target = Exited_Bool  (classification MI)
     │  2. target = NumOfProducts (regression MI)
     │
Feature Engineering & Subset Analysis
     ├── HasBalance = (Balance > 0).astype(int)
     └── re-run MI on Balance > 0 subset
```

**Zero-inflation insight**: ~30–40% of Balance values are zero, dominating the MI signal. `HasBalance` binary flag captures the dominant signal. Subset analysis (Balance > 0) reveals true relationships for active accounts. Core lesson: visualise before engineering.

---

## Datasets at a Glance

| Dataset | File | Rows | Features | Target | Task |
|---------|------|------|----------|--------|------|
| UCI Adult Income | `adult.csv` | ~30 000 | 14 | income (binary) | Classification |
| Diabetes | `Naive-Bayes-Classification-Data.csv` | ~2 000 | 2 | diabetes (binary) | Classification |
| Bank Churn | `Bank_Churn.csv` | ~10 000 | 12 | Exited (binary) | Classification / Clustering |
| Developer Salary | `train.csv` / `test.csv` | ~varies | 6 | salary_usd (continuous) | Regression |
| Ames Housing | `ames.csv` | ~1 460 | 80 | SalePrice (continuous) | (reference) |

---

## Mathematical Reference

### Distance Metrics

| Metric | Formula | Used in |
|--------|---------|---------|
| Euclidean | $\sqrt{\sum_j(x_j-x_{ij})^2}$ | KNN |
| Manhattan | $\sum_j\|x_j-x_{ij}\|$ | Alternative for KNN |

### Information-Theoretic

| Concept | Formula |
|---------|---------|
| Entropy | $H(Y)=-\sum_c p_c\log p_c$ |
| Mutual Information | $I(X;Y)=H(Y)-H(Y\|X)$ |
| Conditional Entropy | $H(Y\|X)=\sum_x p(x)H(Y\|X=x)$ |

### Regression

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MSE | $\frac{1}{n}\sum(y_i-\hat y_i)^2$ | Penalises large errors heavily |
| RMSE | $\sqrt{\text{MSE}}$ | Same unit as target |
| MAE | $\frac{1}{n}\sum\|y_i-\hat y_i\|$ | Robust to outliers |
| R² | $1-\frac{\text{SS}_\text{res}}{\text{SS}_\text{tot}}$ | Fraction of variance explained |

### Classification

| Metric | Formula |
|--------|---------|
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 | 2 · Precision · Recall / (Precision + Recall) |
| ROC-AUC | Area under TPR vs FPR curve |

---

## System Design Overview

Raw CSV → EDA (distributions, MI, correlations) → Stratified train/test split → Feature engineering → BinaryEncoder/MinMaxScaler (fit on train only) → Model training (Bayes, KNN, RF, Lasso, K-Means) → StratifiedKFold CV hyperparameter tuning → Final evaluation on held-out test → ROC curves, confusion matrices, feature importance plots → Power BI / matplotlib visualisation.

### Model Selection Guide

| Scenario | Recommended Model | Reason |
|----------|------------------|--------|
| Numeric features only, small dataset | Gaussian Naive Bayes | Closed-form, interpretable, fast |
| Mixed features, need probability scores | KNN (k tuned by CV) | Non-parametric, no assumptions |
| High-cardinality categoricals, imbalanced | Random Forest | Built-in feature selection, class weights |
| Regression with many weak features | Lasso | Automatic feature pruning via L1 |
| Regression with complex interactions | Random Forest Regressor | No scaling, captures non-linearity |
| Unsupervised customer profiling | K-Means | Scalable, interpretable centroids |

### Scaling Rules

| Model Type | Needs Scaling? | Why |
|------------|---------------|-----|
| KNN | Yes (MinMaxScaler) | Distance-based — magnitude matters |
| Linear / Lasso Regression | Yes (MinMaxScaler) | Coefficient comparability, solver convergence |
| Decision Trees / Random Forest | No | Splits depend on rank order, not magnitude |
| Naive Bayes (Gaussian) | No | Estimates own per-feature Gaussians |
| K-Means | Yes (MinMaxScaler) | Distance-based — same reason as KNN |

---

*This repository demonstrates that rigorous ML engineering is as much about what you don't do (fitting scalers on the test set, using accuracy on imbalanced data, touching the test set during tuning) as what you do.*
