import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("Bank_Churn_edited.csv")
X=df.copy()
X=X.drop(["Exited","Exited_Bool"], axis=1)
Y=df["Exited_Bool"]

X_clean = X.drop(["CustomerId", "Surname", "PerPerAge", "PerPerGender"], axis=1)

for colname in X_clean.select_dtypes(["object"]):
    X_clean[colname], _=X_clean[colname].factorize()
discrete_features = X_clean.dtypes == int
print(discrete_features.head)

miscores=mutual_info_classif(X=X_clean, y=Y, discrete_features=discrete_features)
mi_score=pd.Series(miscores, index=X_clean.columns, name="MISCORES")
mi_score=mi_score.sort_values(ascending=False)
print(mi_score)

plt.figure(figsize=(10,6))
sns.barplot(x=mi_score.values, y=mi_score.index)
plt.xticks(rotation=45, ha="right")
plt.tight_layout
plt.show()