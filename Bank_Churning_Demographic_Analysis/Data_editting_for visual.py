import pandas as pd
df=pd.read_csv("Bank_Churn.csv", na_values="?")
X=df.copy()
X["Exited_Bool"]=X["Exited"].astype("bool")
print(X.head())
X["PerPerAge"]=X.groupby("Age").Exited_Bool.transform("mean")*100
print(X.head())
X["PerPerGender"]=X.groupby("Gender").Exited_Bool.transform("mean")*100
print(X.head())
X = X.astype({
    "EstimatedSalary": "float32",
    "Balance": "float32",
    "CreditScore": "float32"
})

X.to_csv('Bank_Churn_edited.csv', index=False)
