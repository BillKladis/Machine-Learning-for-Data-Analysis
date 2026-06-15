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

Binary classification: predict whether an individual's annual income exceeds \$50 K based on census attributes.

**Dataset** — UCI Adult Census Income (`adult.csv`, ~30 000 rows, 15 features):

| Feature | Type | Description |
|---------|------|-------------|
| age | Numeric | Age in years |
| workclass | Categorical | Employment type (Private, Gov, etc.) |
| education | Categorical | Highest level attained |
| occupation | Categorical | Job category |
| relationship | Categorical | Family role |
| race, gender | Categorical | Demographic attributes |
| native-country | Categorical | Country of origin |
| capital-gain / loss | Numeric | Investment income/loss |
| hours-per-week | Numeric | Average working hours |
| income | Binary target | <=50K or >50K |

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

k is swept from 1 to 31 (odd values prevent ties). For each k, 5-fold stratified CV computes mean ROC-AUC on the training set. The optimal k is selected at the peak of this curve.

**Result**: k = 29, ROC-AUC ≈ 0.88

Small k → high variance (overfits individual noise points). Large k → high bias (smooths decision boundary too aggressively). k = 29 sits in the sweet-spot for this dataset.

### Evaluation

| Metric | Value |
|--------|-------|
| ROC-AUC | ~0.88 |
| Confusion matrix | See `plot_confusion_matrix.png` |
| Precision / Recall | See `plot_precision_recall.png` |

**Outputs**: `plot_k_tuning.png`, `plot_roc_curve.png`, `plot_confusion_matrix.png`, `plot_precision_recall.png`

---

## 2. Naive Bayes — Diabetes Classification

**Location**: `Bayes/Naive-Bayes.py`

### Task

Binary classification: predict whether a patient has diabetes (0 / 1) given glucose level and blood pressure.

**Dataset** (`Naive-Bayes-Classification-Data.csv`):

| Feature | Type |
|---------|------|
| glucose | Numeric |
| bloodpressure | Numeric |
| diabetes | Binary target (0 = No, 1 = Yes) |

### Mathematical Foundation — Gaussian Naive Bayes

Naive Bayes applies Bayes' theorem with the **conditional independence** assumption among features given the class:

$$P(y \mid \mathbf{x}) \propto P(y) \prod_{j=1}^{p} P(x_j \mid y)$$

For continuous features the likelihood is modelled as a Gaussian:

$$P(x_j \mid y = c) = \frac{1}{\sqrt{2\pi\sigma_{cj}^2}} \exp\!\left(-\frac{(x_j - \mu_{cj})^2}{2\sigma_{cj}^2}\right)$$

where μ_cj and σ²_cj are estimated as the sample mean and variance of feature j within class c on the training data.

**Prediction**:

$$\hat{y} = \arg\max_{c}\left[\log P(y=c) + \sum_{j=1}^{p} \log P(x_j \mid y=c)\right]$$

Working in log-space avoids numerical underflow when multiplying many small probabilities.

### Feature Importance (Two Methods)

**Method 1 — Mean difference**: measures how much each feature's class-conditional mean separates the classes.

$$\text{importance}_j = |\mu_{1j} - \mu_{0j}|$$

Implemented as `gnb.theta_[1] - gnb.theta_[0]` (theta stores class means).

**Method 2 — Log-likelihood gap**: for each feature, computes the average log-likelihood ratio between the positive and negative class over the training data. A larger gap means the feature provides more discriminative signal.

$$\text{importance}_j = \mathbb{E}\left[\log P(x_j \mid y=1) - \log P(x_j \mid y=0)\right]$$

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

| Metric | Value |
|--------|-------|
| ROC-AUC | ~0.91 |

RF outperforms KNN (~0.88) because it builds non-linear, axis-aligned decision boundaries in high-dimensional space without distance-based sensitivity to scale.

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

$$\hat{y} = \mathbf{x}^{\top}\boldsymbol{\beta} = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p$$

Coefficients minimise mean squared error (MSE):

$$\mathcal{L}(\boldsymbol{\beta}) = \frac{1}{n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2$$

Requires **MinMaxScaler** preprocessing: gradient-based solvers converge faster and coefficient magnitudes are comparable when features share the same scale.

#### Lasso Regression

Lasso adds an L1 penalty to the linear regression loss:

$$\mathcal{L}(\boldsymbol{\beta}) = \frac{1}{n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda\|\boldsymbol{\beta}\|_1$$

The L1 term drives weak coefficients exactly to zero, performing **automatic feature selection**. The regularisation strength λ is chosen via `LassoCV` (5-fold CV over a log-spaced grid), eliminating the need for a manual grid search.

**Key insight**: unlike Ridge (L2 penalty), the L1 penalty produces a non-differentiable corner at β_j = 0, which induces sparsity. Features whose signal-to-noise ratio is too low receive a zero coefficient and are effectively removed.

#### Random Forest Regressor

Same ensemble mechanism as the classifier but predicts the **mean** of the leaf values rather than the majority class:

$$\hat{y} = \frac{1}{T}\sum_{t=1}^{T} h_t(\mathbf{x})$$

Splitting criterion: minimise the weighted variance of child nodes.

**No scaling needed**: tree splits depend only on feature rank order, not absolute magnitude.

### Evaluation Metrics

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

R² measures explained variance (1.0 = perfect, 0.0 = predicts the mean). RMSE is in the same units as the target (USD), making it directly interpretable.

**Residual analysis** — plotting (ŷ − y) vs ŷ reveals heteroscedasticity, non-linearity, and influential outliers that summary statistics miss.

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

### Discrete vs. Continuous Features

The function accepts a `discrete_features` boolean mask. Integer-coded categorical columns use the discrete estimator (plug-in MI); floating-point columns use the continuous estimator. Misclassifying a continuous feature as discrete inflates MI by binning it.

**Supplementary plots**: boxplot of blood pressure by diabetes class, scatter (glucose vs. blood pressure coloured by class), correlation heatmap — together these provide complementary linear and distributional views alongside the non-linear MI ranking.

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

### Script 1 — Data Preprocessing (`Data_editting_for visual.py`)

Prepares the dataset for Power BI visualisation:

- **`Exited_Bool`**: boolean cast of `Exited` for clean filtering in BI tools.
- **`PerPerAge`** and **`PerPerGender`**: pre-aggregated churn rates (%) by age group and gender, reducing load on the BI engine.
- Type casting (`float32`): reduces memory from 64-bit defaults, important when feeding large tables into Power BI.

Output: `Bank_Churn_edited.csv` → `PowerBI.csv`.

### Script 2 — K-Means Customer Segmentation (`KMeans_feature_eng.py`)

**Features used**: Balance, EstimatedSalary, CreditScore (scaled with MinMaxScaler).

**K-Means objective** — minimise within-cluster sum of squared distances:

$$J = \sum_{k=1}^{K}\sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|_2^2$$

**Algorithm (Lloyd's algorithm)**:
1. Initialise K centroids (k-means++ heuristic for spread initialisation).
2. Assign each point to the nearest centroid.
3. Recompute centroids as cluster means.
4. Repeat until convergence.

Parameters: `n_clusters=6`, `n_init=50` (50 random restarts, keeps the best J to avoid local optima).

**Why 6 clusters?** Experimented to identify financially distinct customer personas. Each cluster is profiled by its mean values and churn rate, surfacing segments like "high-balance low-activity churners" vs. "low-balance loyal customers".

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

**Zero-inflation in Balance**: approximately 30–40% of customers have Balance = 0. This bimodal structure means:
- The raw Balance → NumOfProducts MI is dominated by the zero-mass spike.
- `HasBalance` captures whether a customer *has* any balance, which turns out to be the dominant signal.
- Restricting to Balance > 0 and re-running MI reveals the true relationship for active accounts.

This demonstrates the principle of **distributional awareness before feature engineering**: visualise first, then decide whether to transform or segment.

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

```
┌─────────────────────────────────────────────────────────────┐
│                      Raw Data Sources                        │
│  CSV files (Adult, Diabetes, BankChurn, Salary, Ames)       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Exploratory Analysis                        │
│  • Distribution plots (histograms, KDE)                      │
│  • Correlation matrices                                       │
│  • Mutual Information rankings                                │
│  • Zero-inflation checks (Bank Balance)                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Preprocessing Layer                          │
│  ┌─────────────────┐   ┌───────────────────────────────┐    │
│  │ train_test_split │   │ Feature Engineering           │    │
│  │ (stratified 80/20│   │ • count_items() for lists     │    │
│  │  or KFold CV)   │   │ • HasBalance binary indicator  │    │
│  └────────┬────────┘   │ • churn rates per demographic │    │
│           │             └───────────────────────────────┘    │
│           ▼                                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Encoders / Scalers (fit on TRAIN only)             │    │
│  │  BinaryEncoder → categorical columns               │    │
│  │  MinMaxScaler  → numeric columns (distance models)  │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Training Layer                       │
│                                                               │
│  ┌──────────┐ ┌─────────────┐ ┌───────────────────────┐    │
│  │ Gaussian │ │  KNN (k=29) │ │   Random Forest        │    │
│  │ Naive    │ │  Euclidean  │ │   (Bagging + Random    │    │
│  │ Bayes    │ │  Distance   │ │    Subspaces)          │    │
│  └──────────┘ └─────────────┘ └───────────────────────┘    │
│                                                               │
│  ┌────────────────┐ ┌───────────────┐ ┌────────────────┐   │
│  │ Linear         │ │ Lasso         │ │ K-Means        │   │
│  │ Regression     │ │ (L1 + LassoCV)│ │ (Segmentation) │   │
│  └────────────────┘ └───────────────┘ └────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Hyperparameter Tuning (CV Loop)                  │
│  StratifiedKFold (5-fold) on train set only                  │
│  Metric: ROC-AUC (classification) / R² (regression)         │
│  Grid: k ∈ [1,31], n_est ∈ [50,350], λ auto via LassoCV    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Evaluation Layer                           │
│  • Final metrics on held-out TEST set (touched once)        │
│  • ROC curve, Precision-Recall curve                         │
│  • Confusion matrix                                           │
│  • Residual plots (regression)                               │
│  • Feature importance plots                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Reporting & Visualisation                   │
│  matplotlib / seaborn PNGs  ←→  Power BI (.pbix, .csv)     │
└─────────────────────────────────────────────────────────────┘
```

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
