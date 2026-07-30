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

# =========================
# Load Dataset
# =========================

df = pd.read_csv("heart.csv")

# Create models folder
os.makedirs("models", exist_ok=True)

# =========================
# Input and Target
# =========================

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# =========================
# One Hot Encoding
# =========================

X = pd.get_dummies(X, drop_first=True)

# Save column names
columns = X.columns.tolist()

# =========================
# Feature Scaling
# =========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Save Scaler & Columns
# =========================

joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(columns, "models/columns.pkl")

# =========================
# Models
# =========================

models = {

    "LogisticRegression.pkl": LogisticRegression(max_iter=1000),

    "DecisionTreeClassifier.pkl": DecisionTreeClassifier(),

    "SVM.pkl": SVC(),

    "KNN.pkl": KNeighborsClassifier(),

    "NaiveBayes.pkl": GaussianNB()

}

# =========================
# Train & Save Models
# =========================

for filename, model in models.items():

    model.fit(X_train, y_train)

    joblib.dump(model, f"models/{filename}")

print("Classification Models Saved Successfully!")