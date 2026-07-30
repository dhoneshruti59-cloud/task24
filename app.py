import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Multi Model Prediction App", layout="centered")

st.title("🏥 Heart Disease & Insurance Prediction")

problem = st.selectbox(
    "Select Problem Type",
    ["Classification", "Regression"]
)

# ==========================================================
# CLASSIFICATION
# ==========================================================

if problem == "Classification":

    model_name = st.selectbox(
        "Select Classification Model",
        [
            "LogisticRegression",
            "DecisionTreeClassifier",
            "SVM",
            "KNN",
            "NaiveBayes"
        ]
    )

    model = joblib.load(f"models/{model_name}.pkl")
    scaler = joblib.load("models/scaler.pkl")
    columns = joblib.load("models/columns.pkl")

    st.header("Enter Patient Details")

    Age = st.number_input("Age", 1, 120, 30)

    Sex = st.selectbox("Sex", ["M", "F"])

    ChestPainType = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "ASY", "TA"]
    )

    RestingBP = st.number_input(
        "Resting BP",
        50,
        250,
        120
    )

    Cholesterol = st.number_input(
        "Cholesterol",
        0,
        700,
        200
    )

    FastingBS = st.selectbox(
        "Fasting Blood Sugar",
        [0,1]
    )

    RestingECG = st.selectbox(
        "Resting ECG",
        ["Normal","ST","LVH"]
    )

    MaxHR = st.number_input(
        "Max Heart Rate",
        50,
        250,
        150
    )

    ExerciseAngina = st.selectbox(
        "Exercise Angina",
        ["Y","N"]
    )

    Oldpeak = st.number_input(
        "Oldpeak",
        0.0,
        10.0,
        1.0
    )

    ST_Slope = st.selectbox(
        "ST Slope",
        ["Up","Flat","Down"]
    )

    user_data = pd.DataFrame({

        "Age":[Age],
        "Sex":[Sex],
        "ChestPainType":[ChestPainType],
        "RestingBP":[RestingBP],
        "Cholesterol":[Cholesterol],
        "FastingBS":[FastingBS],
        "RestingECG":[RestingECG],
        "MaxHR":[MaxHR],
        "ExerciseAngina":[ExerciseAngina],
        "Oldpeak":[Oldpeak],
        "ST_Slope":[ST_Slope]

    })

    user_data = pd.get_dummies(user_data, drop_first=True)

    user_data = user_data.reindex(
        columns=columns,
        fill_value=0
    )

    user_data = scaler.transform(user_data)

    if st.button("Predict"):

        prediction = model.predict(user_data)

        if prediction[0] == 1:
            st.success("Heart Disease Present")
        else:
            st.success("No Heart Disease")

# ==========================================================
# REGRESSION
# ==========================================================

else:

    model_name = st.selectbox(
        "Select Regression Model",
        [
            "LinearRegression",
            "DecisionTreeRegressor",
            "SVR",
            "KNNRegressor"
        ]
    )

    model = joblib.load(f"models/{model_name}.pkl")
    scaler = joblib.load("models/reg_scaler.pkl")
    columns = joblib.load("models/reg_columns.pkl")

    st.header("Insurance Details")

    age = st.number_input(
        "Age",
        18,
        100,
        25
    )

    sex = st.selectbox(
        "Sex",
        ["male","female"]
    )

    bmi = st.number_input(
        "BMI",
        10.0,
        60.0,
        25.0
    )

    children = st.number_input(
        "Children",
        0,
        10,
        0
    )

    smoker = st.selectbox(
        "Smoker",
        ["yes","no"]
    )

    region = st.selectbox(
        "Region",
        [
            "northeast",
            "northwest",
            "southeast",
            "southwest"
        ]
    )

    user_data = pd.DataFrame({

        "age":[age],
        "sex":[sex],
        "bmi":[bmi],
        "children":[children],
        "smoker":[smoker],
        "region":[region]

    })

    user_data = pd.get_dummies(
        user_data,
        drop_first=True
    )

    user_data = user_data.reindex(
        columns=columns,
        fill_value=0
    )

    user_data = scaler.transform(user_data)

    if st.button("Predict"):

        prediction = model.predict(user_data)

        st.success(
            f"Predicted Insurance Expense : ₹ {prediction[0]:.2f}"
        )