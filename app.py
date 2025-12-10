
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model, scaler, and X_train columns
@st.cache_resource
def load_model_artifacts():
    final_model = joblib.load('best_rf_heart_disease_model.joblib')
    scaler = joblib.load('scaler.joblib')
    X_train_cols = joblib.load('X_train_columns.joblib')
    return final_model, scaler, X_train_cols

final_model, scaler, X_train_cols = load_model_artifacts()

# Define original feature columns
original_categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'FastingBS']
original_numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

# Validation ranges
VALIDATION_RANGES = {
    'Age': (1, 100),
    'RestingBP': (50, 250),
    'Cholesterol': (50, 700),
    'MaxHR': (50, 220),
    'Oldpeak': (-3.0, 7.0)
}

# Streamlit layout
st.set_page_config(layout="wide")
st.title("Heart Disease Prediction App")
st.write("Enter patient details to predict heart disease likelihood.")

# Sidebar inputs
st.sidebar.header("Patient Input Features")

def user_input_features():
    age = st.sidebar.slider('Age', VALIDATION_RANGES['Age'][0], VALIDATION_RANGES['Age'][1], 54)
    sex = st.sidebar.selectbox('Sex', ('M', 'F'))
    chest_pain_type = st.sidebar.selectbox('ChestPainType', ('ATA', 'NAP', 'ASY', 'TA'))
    resting_bp = st.sidebar.slider('RestingBP (mm Hg)', VALIDATION_RANGES['RestingBP'][0], VALIDATION_RANGES['RestingBP'][1], 130)
    cholesterol = st.sidebar.slider('Cholesterol', VALIDATION_RANGES['Cholesterol'][0], VALIDATION_RANGES['Cholesterol'][1], 223)
    fasting_bs = st.sidebar.selectbox('FastingBS (>120 mg/dl)', (0, 1))
    resting_ecg = st.sidebar.selectbox('RestingECG', ('Normal', 'ST', 'LVH'))
    max_hr = st.sidebar.slider('MaxHR (bpm)', VALIDATION_RANGES['MaxHR'][0], VALIDATION_RANGES['MaxHR'][1], 138)
    exercise_angina = st.sidebar.selectbox('ExerciseAngina', ('N', 'Y'))
    oldpeak = st.sidebar.slider('Oldpeak', float(VALIDATION_RANGES['Oldpeak'][0]),
                                float(VALIDATION_RANGES['Oldpeak'][1]), 0.6, step=0.1)
    st_slope = st.sidebar.selectbox('ST_Slope', ('Up', 'Flat', 'Down'))

    data = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain_type,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader('User Input Features')
st.write(input_df)

# ===========================
#       PREDICT BUTTON
# ===========================
if st.button('Predict Heart Disease'):
    try:
        processed_input = input_df.copy()

        # One-hot encode
        processed_input = pd.get_dummies(processed_input, columns=original_categorical_cols)

        # Align columns
        processed_input = processed_input.reindex(columns=X_train_cols, fill_value=0)

        # Convert bool to int
        for col in processed_input.columns:
            if processed_input[col].dtype == 'bool':
                processed_input[col] = processed_input[col].astype(int)

        # Scale numerical features
        processed_input[original_numerical_cols] = scaler.transform(
            processed_input[original_numerical_cols]
        )

        # ----------------------------
        #        SAFE PROBA FIX
        # ----------------------------
        prediction = final_model.predict(processed_input)
        proba = final_model.predict_proba(processed_input)

        if proba.shape[1] == 1:
            # Model only output probability of class 0 → convert to class 1 probability
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

        # ===========================
        #        SHAP SECTION
        # ===========================
        st.subheader("Explanation of Prediction (SHAP Values)")

        explainer = shap.Explainer(final_model, processed_input)
        shap_values = explainer(processed_input)

        st.write("**Waterfall Plot:**")
        fig_wf, ax_wf = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig_wf)

        st.write("**Feature Importance (Bar Plot):**")
        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        shap.plots.bar(shap_values, show=False)
        st.pyplot(fig_bar)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
