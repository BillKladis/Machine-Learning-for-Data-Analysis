import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import seaborn as sns
import matplotlib.pyplot as plt
#---------------- Data Loading and Preprocessing(No need for data cleaning or transformation) ----------------

df=pd.read_csv("Naive-Bayes-Classification-Data.csv", header=0, na_values=["?"])
X=df.drop("diabetes", axis=1)
Y=df["diabetes"]
print(X.head())

for colname in X.select_dtypes(["object"]):
    X[colname], _= X[colname].factorize()
discrete_features = X.dtypes == int
print(discrete_features.head)

mi_score=mutual_info_classif(X,Y, discrete_features=discrete_features)
mi_scores=pd.Series(mi_score, name="MI Score", index=X.columns)
mi_scores=mi_scores.sort_values(ascending=False)
print(mi_scores)


plt.figure(figsize=(10, 6))
sns.barplot(y=mi_scores.index, x=mi_scores.values)
plt.xticks(rotation=45, ha="right")
plt.title("Mutual Information Scores")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x="diabetes", y="bloodpressure", data=df, width=0.4)
plt.title("Blood Pressure by Diabetes Status")
plt.xlabel("Diabetes (0 = No, 1 = Yes)")
plt.ylabel("Blood Pressure")
plt.tight_layout()
plt.show()


sns.lmplot(x="glucose", y="bloodpressure", data=df)
plt.title("Blood Pressure vs Glucose")
plt.xlabel("Glucose")
plt.ylabel("Blood Pressure")
plt.tight_layout()
plt.show()

print(df[["glucose", "bloodpressure"]].corr())
