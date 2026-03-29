import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("Bank_Churn_edited.csv")
print(df.head())
features=["Balance","EstimatedSalary","CreditScore"]
X=df[features]

scaler=MinMaxScaler()
X_scaled_array=scaler.fit_transform(X)
X = pd.DataFrame(X_scaled_array, columns=features, index=X.index)
print(X.head())
kmeans=KMeans(n_clusters=6, n_init=20)

X["Cluster"]=kmeans.fit_predict(X)

Display=df.copy()
Display["Cluster"]=X["Cluster"]
average_features_cluster=Display.groupby("Cluster")[features].mean()

cluster_means=X.groupby("Cluster")[features].mean()

Xy = X.copy()
Xy["Cluster"] = Xy.Cluster.astype("category")
Xy["Exited_Bool"] = df["Exited_Bool"]
plt.figure(figsize=(10, 4))
sns.heatmap(cluster_means, annot=average_features_cluster, fmt=",.0f", cmap="YlGnBu")
plt.title("Cluster Profiles (Actual Averages shown in cells)")
plt.ylabel("Cluster Number")
plt.show()

churn_data = Display.groupby("Cluster")["Exited_Bool"].mean().reset_index()
churn_data1=Display.groupby("Cluster")["Exited_Bool"].mean()

print("no sort index\n",churn_data1.head())
print(" sort index\n",churn_data.head())

plt.figure(figsize=(10,4))
sns.barplot(data=churn_data, x="Cluster", y="Exited_Bool", palette="YlGnBu")
plt.title("Avg Churn per Cluster")
plt.ylabel("Churn_Rate")
plt.xlabel("Cluster")
plt.show()
print(df.head())

