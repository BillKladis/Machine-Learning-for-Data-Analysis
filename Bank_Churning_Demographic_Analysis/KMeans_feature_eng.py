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
kmeans=KMeans(n_clusters=6, n_init=50, random_state=42)

X["Cluster"]=kmeans.fit_predict(X)

Display=df.copy()
Display["Cluster"]=X["Cluster"]
average_features_cluster=Display.groupby("Cluster")[features].mean()

cluster_means=X.groupby("Cluster")[features].mean()

Xy = X.copy()
Xy["Cluster"] = Xy.Cluster.astype("category")
Xy["Exited_Bool"] = df["Exited_Bool"]

churn_data = Display.groupby("Cluster")["Exited_Bool"].mean().reset_index()
churn_data1=Display.groupby("Cluster")["Exited_Bool"].mean()

print("no sort index\n",churn_data1.head())
print(" sort index\n",churn_data.head())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(cluster_means, annot=average_features_cluster, fmt=",.0f", 
            cmap="YlGnBu", ax=ax1)
ax1.set_title("Customer Profiles")
sns.barplot(data=churn_data, x="Cluster", y="Exited_Bool", hue="Cluster", 
            palette="YlGnBu", ax=ax2)
ax2.set_title("Churn Rate per Cluster")
plt.tight_layout()
plt.show()
print(df.head())

