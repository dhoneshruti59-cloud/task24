import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv("heart.csv")

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X = pd.get_dummies(X, drop_first=True)

columns = X.columns

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

os.makedirs("models", exist_ok=True)

joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(columns, "models/columns.pkl")


models = {
    "LogisticRegression.pkl": LogisticRegression(max_iter=1000),
    "DecisionTreeClassifier.pkl": DecisionTreeClassifier(),
    "SVM.pkl": SVC(),
    "KNN.pkl": KNeighborsClassifier(),
    "NaiveBayes.pkl": GaussianNB()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, "models/" + name)

print("Classification Models Saved")

X_reg = df.drop("Cholesterol", axis=1)
y_reg = df["Cholesterol"]

X_reg = pd.get_dummies(X_reg, drop_first=True)

scaler_reg = StandardScaler()
X_reg = scaler_reg.fit_transform(X_reg)

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

joblib.dump(scaler_reg, "models/reg_scaler.pkl")
joblib.dump(X_reg.shape[1], "models/reg_columns.pkl")

reg_models = {
    "LinearRegression.pkl": LinearRegression(),
    "DecisionTreeRegressor.pkl": DecisionTreeRegressor(),
    "SVR.pkl": SVR(),
    "KNNRegressor.pkl": KNeighborsRegressor()
}

for name, model in reg_models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, "models/" + name)

print("Regression Models Saved")