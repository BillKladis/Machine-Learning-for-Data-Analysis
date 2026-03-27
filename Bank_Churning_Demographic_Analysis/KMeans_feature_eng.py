import pandas as pd
from sklearn.cluster import KMeans
df=pd.read_csv("Bank_Churn_edited.csv")
print(df.head())
X=df[["Balance","EstimatedSalary","CreditScore"]]
kmeans=KMeans(n_clusters=4, n_init=20)
X["Cluster"]=kmeans.fit_predict(X)
X["Cluster"] = X["Cluster"].astype("category")


print(df.head())

