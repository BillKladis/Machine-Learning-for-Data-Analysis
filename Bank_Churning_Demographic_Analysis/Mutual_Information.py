import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# Set a consistent plotting style
sns.set_theme(style="whitegrid")

def load_and_clean_data(file_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Loads the dataset, separates the target variable, and cleans the features.
    """
    df = pd.read_csv(file_path)
    
    # Separate features and target
    X = df.drop(columns=["Exited", "Exited_Bool"], errors="ignore")
    y = df["Exited_Bool"]
    
    # Drop irrelevant columns
    cols_to_drop = ["CustomerId", "Surname", "PerPerAge", "PerPerGender"]
    X_clean = X.drop(columns=cols_to_drop, errors="ignore")
    
    # Factorize object-type columns
    for colname in X_clean.select_dtypes(include=["object"]).columns:
        X_clean[colname], _ = X_clean[colname].factorize()
        
    return X, X_clean, y

def calculate_mi_scores(X: pd.DataFrame, y: pd.Series, task: str = "classification") -> pd.Series:
    """
    Calculates mutual information scores for either classification or regression tasks.
    """
    if task == "classification":
        # Assume integer columns are discrete features
        discrete_features = X.dtypes == int
        mi_scores = mutual_info_classif(
            X=X, y=y, discrete_features=discrete_features, random_state=42
        )
    elif task == "regression":
        # Average MI scores over a couple of random states for stability
        all_scores = []
        for i in range(2):
            s = mutual_info_regression(X, y, random_state=i)
            all_scores.append(s)
        mi_scores = np.mean(all_scores, axis=0)
    else:
        raise ValueError("Task must be either 'classification' or 'regression'.")
        
    mi_series = pd.Series(mi_scores, index=X.columns, name="MI_Scores")
    return mi_series.sort_values(ascending=False)

def plot_mi_scores(mi_scores: pd.Series, title: str) -> None:
    """
    Generates a horizontal barplot for Mutual Information scores.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=mi_scores.values, y=mi_scores.index, palette="viridis", hue=mi_scores.index, legend=False)
    plt.title(title)
    plt.xlabel("Mutual Information Score")
    plt.ylabel("Features")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def run_exploratory_data_analysis(X_original: pd.DataFrame, X_clean: pd.DataFrame) -> None:
    """
    Runs basic EDA plotting and correlation checks.
    """
    # Scatter plot: Age vs NumOfProducts
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=X_clean["Age"], y=X_clean["NumOfProducts"], alpha=0.6)
    plt.title("Age vs Number of Products")
    plt.tight_layout()
    plt.show()
    
    correlation = X_original["Age"].corr(X_clean["NumOfProducts"])
    print(f"Correlation between Age and NumOfProducts: {correlation:.4f}")

    # Box plot: NumOfProducts vs Balance
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=X_clean["NumOfProducts"], y=X_clean["Balance"], palette="muted", hue=X_clean["NumOfProducts"], legend=False)
    plt.title("Balance Distribution by Number of Products")
    plt.tight_layout()
    plt.show()

def main():
    # 1. Load Data
    file_path = "Bank_Churn_edited.csv"
    X_original, X_clean, y_target = load_and_clean_data(file_path)
    
    print("Discrete Features Check:")
    print((X_clean.dtypes == int).head())
    print("-" * 40)

    # 2. Classification MI Scores (Target: Exited_Bool)
    mi_scores_class = calculate_mi_scores(X_clean, y_target, task="classification")
    print("Classification MI Scores (Target: Exited_Bool):")
    print(mi_scores_class)
    print("-" * 40)
    plot_mi_scores(mi_scores_class, "Mutual Information Scores (Classification)")

    # 3. Exploratory Data Analysis
    run_exploratory_data_analysis(X_original, X_clean)

    # 4. Regression MI Scores (Target: NumOfProducts)
    y_products = X_clean["NumOfProducts"]
    X_mi_regression = X_clean.drop(columns=["NumOfProducts"])
    
    mi_scores_reg = calculate_mi_scores(X_mi_regression, y_products, task="regression")
    print("Regression MI Scores (Target: NumOfProducts):")
    print(mi_scores_reg)
    print("-" * 40)
    plot_mi_scores(mi_scores_reg, "Mutual Information Scores (Regression)")

    # 5. Feature Engineering and Follow-up MI Tests
    # Note: Creating a binary 'HasBalance' feature to test its specific impact
    X_mi_regression["HasBalance"] = (X_mi_regression["Balance"] > 0).astype(int)
    
    mi_has_balance = mutual_info_classif(X=X_mi_regression[["HasBalance"]], y=y_products, random_state=42)
    mi_raw_balance = mutual_info_classif(X=X_mi_regression[["Balance"]], y=y_products, random_state=42)
    
    print(f"MI Score (HasBalance -> NumOfProducts): {mi_has_balance[0]:.6f}")
    print(f"MI Score (Raw Balance -> NumOfProducts): {mi_raw_balance[0]:.6f}")

if __name__ == "__main__":
    main()