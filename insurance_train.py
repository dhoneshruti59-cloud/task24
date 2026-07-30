import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# =========================
# Load Dataset
# =========================

df = pd.read_csv("insurance_1.csv")

# Create models folder
os.makedirs("models", exist_ok=True)

# =========================
# Input and Target
# =========================

X = df.drop("expenses", axis=1)
y = df["expenses"]

# =========================
# One Hot Encoding
# =========================

X = pd.get_dummies(X, drop_first=True)

# Save Columns
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

joblib.dump(scaler, "models/reg_scaler.pkl")
joblib.dump(columns, "models/reg_columns.pkl")

# =========================
# Models
# =========================

models = {

    "LinearRegression.pkl": LinearRegression(),

    "DecisionTreeRegressor.pkl": DecisionTreeRegressor(),

    "SVR.pkl": SVR(),

    "KNNRegressor.pkl": KNeighborsRegressor()

}

# =========================
# Train & Save Models
# =========================

for filename, model in models.items():

    model.fit(X_train, y_train)

    joblib.dump(model, f"models/{filename}")

print("Regression Models Saved Successfully!")