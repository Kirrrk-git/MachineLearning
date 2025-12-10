import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ---------------------------
# Load model + scaler + columns
# ---------------------------
@st.cache_resource
def load_model_artifacts():
    final_model = joblib.load("best_rf_heart_disease_model.joblib")
    scaler = joblib.load("scaler.joblib")
    X_train_cols = joblib.load("X_train_columns.joblib")
    return final_model, scaler, X_train_cols

final_model, scaler, X_train_cols = load_model_artifacts()


# ---------------------------
# Feature groups
# ---------------------------
original_categorical_cols = [
    "Sex", "ChestPainType", "RestingECG",
    "ExerciseAngina", "ST_Slope", "FastingBS"
]

original_numerical_cols = [
    "Age", "RestingBP", "Cholesterol",
    "MaxHR", "Oldpeak"
]

VALIDATION_RANGES = {
    "Age": (1, 100),
    "RestingBP": (50, 250),
    "Cholesterol": (50, 700),
    "MaxHR": (50, 220),
    "Oldpeak": (-3.0, 7.0)
}

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(layout="wide")
st.title("Heart Disease Prediction App")
st.write("Enter patient details to predict the likelihood of heart disease.")

st.sidebar.header("Patient Input Features")


def user_input_features():
    age = st.sidebar.slider("Age", *VALIDATION_RANGES["Age"], 54)
    sex = st.sidebar.selectbox("Sex", ("M", "F"))
    chest_pain_type = st.sidebar.selectbox("ChestPainType", ("ATA", "NAP", "ASY", "TA"))
    resting_bp = st.sidebar.slider("RestingBP", *VALIDATION_RANGES["RestingBP"], 130)
    cholesterol = st.sidebar.slider("Cholesterol", *VALIDATION_RANGES["Cholesterol"], 223)
    fasting_bs = st.sidebar.selectbox("FastingBS (>120mg/dl)", (0, 1))
    resting_ecg = st.sidebar.selectbox("RestingECG", ("Normal", "ST", "LVH"))
    max_hr = st.sidebar.slider("MaxHR", *VALIDATION_RANGES["MaxHR"], 138)
    exercise_angina = st.sidebar.selectbox("ExerciseAngina", ("N", "Y"))
    oldpeak = st.sidebar.slider("Oldpeak", *VALIDATION_RANGES["Oldpeak"], 0.6)
    st_slope = st.sidebar.selectbox("ST_Slope", ("Up", "Flat", "Down"))

    data = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain_type,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }
    return pd.DataFrame(data, index=[0])


input_df = user_input_features()

st.subheader("User Input Features")
st.write(input_df)


# ---------------------------
# PREDICT BUTTON
# ---------------------------
if st.button("Predict Heart Disease"):
    try:
        # ---------------------------
        # Preprocessing
        # ---------------------------
        processed_input = input_df.copy()

        processed_input = pd.get_dummies(processed_input, columns=original_categorical_cols)

        # align to training columns
        processed_input = processed_input.reindex(columns=X_train_cols, fill_value=0)

        # convert bool → int
        for col in processed_input.columns:
            if processed_input[col].dtype == "bool":
                processed_input[col] = processed_input[col].astype(int)

        # scale numeric features
        processed_input[original_numerical_cols] = scaler.transform(
            processed_input[original_numerical_cols]
        )

        # ---------------------------
        # Prediction + safe proba fix
        # ---------------------------
        prediction = final_model.predict(processed_input)
        proba = final_model.predict_proba(processed_input)

        # If model outputs only P(class0)
        if proba.shape[1] == 1:
            prediction_proba = 1 - proba[:, 0]
        else:
            prediction_proba = proba[:, 1]

        st.subheader("Prediction")
        result_text = "Presence of Heart Disease" if prediction[0] == 1 else "Absence of Heart Disease"

        if prediction[0] == 1:
            st.warning(f"Prediction: **{result_text}**")
        else:
            st.success(f"Prediction: **{result_text}**")

        st.write(f"Probability of Heart Disease: **{prediction_proba[0]:.2f}**")

        # ---------------------------
        # SHAP EXPLANATION
        # ---------------------------
        st.subheader("Explanation of Prediction (SHAP Values)")

        explainer = shap.Explainer(final_model, processed_input)
        shap_values = explainer(processed_input)

        # SHAP values shape handling
        values = shap_values.values  # can be (1, n_features) or (1, n_features, 2)

        # If model outputs SHAP for two classes → pick the positive class
        if values.ndim == 3 and values.shape[-1] == 2:
            shap_values_class1 = values[0, :, 1]
        else:
            shap_values_class1 = values[0]

        # Build a single explanation object for plotting
        shap_expl = shap.Explanation(
            values=shap_values_class1,
            base_values=shap_values.base_values[0],
            data=processed_input.iloc[0],
            feature_names=processed_input.columns
        )

        # Waterfall Plot
        st.write("**Waterfall Plot:**")
        fig_wf, ax_wf = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_expl, show=False)
        st.pyplot(fig_wf)

        # Bar Plot
        st.write("**Feature Importance (Bar Plot):**")
        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        shap.plots.bar(shap_expl, show=False)
        st.pyplot(fig_bar)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
