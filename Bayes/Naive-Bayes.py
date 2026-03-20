
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import norm
import numpy as np
#---------------- Data Loading and Preprocessing(No need for data cleaning or transformation) ----------------
df=pd.read_csv('Naive-Bayes-Classification-Data.csv', na_values=["?"])
X=df.drop('diabetes', axis=1)
Y = df['diabetes']
#
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
gnb=GaussianNB()
gnb.fit(X_train, Y_train)
#---------------- Performance Evaluation -----------------------------------------------------------------
Y_pred = gnb.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Classification Report:")
print(classification_report(Y_test, Y_pred))
print("Confusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))
#---------------- Feature Importance Analysis (Optional) ------------------------------------------------------
#------------------------------- Using the difference in mean values for each class as a simple measure of feature importance -------------
feature_importance = gnb.theta_[1] - gnb.theta_[0]
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("Feature Importance:")
print(importance_df)
#-------------------------------- Using the average log-likelihood difference as a more nuanced measure of feature importance -------------
log_importance = []
for i in range(X_train.shape[1]):
    # Evaluate each training sample under both class Gaussians
    ll1 = norm.logpdf(X_train.iloc[:, i],
                      loc=gnb.theta_[1, i],          # class-1 mean
                      scale=np.sqrt(gnb.var_[1, i]))  # class-1 std
    ll0 = norm.logpdf(X_train.iloc[:, i],
                      loc=gnb.theta_[0, i],
                      scale=np.sqrt(gnb.var_[0, i]))
    # Mean absolute difference = average discriminative power
    log_importance.append(np.mean(np.abs(ll1 - ll0)))
print("\nLog-Likelihood Based Feature Importance:")
for name, imp in zip(feature_names, log_importance):
    print(f"{name}: {imp:.4f}")