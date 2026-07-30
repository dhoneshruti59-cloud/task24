import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Multi Model Prediction App", layout="centered")

st.title("Classification and Regression prediction App")

problem = st.selectbox(
    "Select Problem Type",
    ["Classification", "Regression"]
)

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

    st.header("Enter Input Values")

    Age = st.number_input("Age", 18, 100, 40)
    RestingBP = st.number_input("RestingBP", 50, 250, 120)
    Cholesterol = st.number_input("Cholesterol", 0, 700, 200)
    FastingBS = st.number_input("FastingBS (0/1)", 0, 1, 0)
    MaxHR = st.number_input("MaxHR", 50, 250, 150)
    Oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)

    Sex = st.selectbox("Sex", ["Male", "Female"])
    ChestPainType = st.selectbox("ChestPainType", ["ATA","NAP","ASY","TA"])
    RestingECG = st.selectbox("RestingECG", ["Normal","ST","LVH"])
    ExerciseAngina = st.selectbox("ExerciseAngina", ["Yes","No"])
    ST_Slope = st.selectbox("ST_Slope", ["Up","Flat","Down"])

    if st.button("Predict Classification"):

        data = pd.DataFrame({
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

        data = pd.get_dummies(data)

        data = data.reindex(columns=columns, fill_value=0)

        data = scaler.transform(data)

        prediction = model.predict(data)

        if prediction[0] == 1:
            st.success("Heart Disease Detected")
        else:
            st.success("No Heart Disease")



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

    st.header("Enter Input Values")

    Age = st.number_input("Age", 18, 100, 40)
    RestingBP = st.number_input("RestingBP", 50, 250, 120)
    FastingBS = st.number_input("FastingBS", 0, 1, 0)
    MaxHR = st.number_input("MaxHR", 50, 250, 150)
    Oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)

    Sex = st.selectbox("Sex ", ["Male","Female"])
    ChestPainType = st.selectbox("ChestPainType ", ["ATA","NAP","ASY","TA"])
    RestingECG = st.selectbox("RestingECG ", ["Normal","ST","LVH"])
    ExerciseAngina = st.selectbox("ExerciseAngina ", ["Yes","No"])
    ST_Slope = st.selectbox("ST_Slope ", ["Up","Flat","Down"])

    if st.button("Predict Regression"):

        data = pd.DataFrame({
            "Age":[Age],
            "Sex":[Sex],
            "ChestPainType":[ChestPainType],
            "RestingBP":[RestingBP],
            "FastingBS":[FastingBS],
            "RestingECG":[RestingECG],
            "MaxHR":[MaxHR],
            "ExerciseAngina":[ExerciseAngina],
            "Oldpeak":[Oldpeak],
            "ST_Slope":[ST_Slope]
        })

        data = pd.get_dummies(data)

        while data.shape[1] < model.n_features_in_:
            data[f"dummy{data.shape[1]}"] = 0

        data = data.iloc[:, :model.n_features_in_]

        data = scaler.transform(data)

        prediction = model.predict(data)

        st.success(f"Predicted Cholesterol : {prediction[0]:.2f}")