%%writefile app.py
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

# Define original feature columns, separated by type as per preprocessing
original_categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'FastingBS']
original_numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

# Define validation ranges for numerical inputs (based on df.describe() from notebook)
VALIDATION_RANGES = {
    'Age': (1, 100),
    'RestingBP': (50, 250),
    'Cholesterol': (50, 700),
    'MaxHR': (50, 220),
    'Oldpeak': (-3.0, 7.0)
}

# Streamlit app layout
st.set_page_config(layout="wide")
st.title("Heart Disease Prediction App")
st.write("Enter patient details to predict the likelihood of heart disease and get explanations.")

# --- Input Widgets ---
st.sidebar.header("Patient Input Features")

def user_input_features():
    age = st.sidebar.slider('Age', VALIDATION_RANGES['Age'][0], VALIDATION_RANGES['Age'][1], 54)
    sex = st.sidebar.selectbox('Sex', ('M', 'F'))
    chest_pain_type = st.sidebar.selectbox('ChestPainType', ('ATA', 'NAP', 'ASY', 'TA'))
    resting_bp = st.sidebar.slider('RestingBP (mm Hg)', VALIDATION_RANGES['RestingBP'][0], VALIDATION_RANGES['RestingBP'][1], 130)
    cholesterol = st.sidebar.slider('Cholesterol (mg/dl)', VALIDATION_RANGES['Cholesterol'][0], VALIDATION_RANGES['Cholesterol'][1], 223)
    fasting_bs = st.sidebar.selectbox('FastingBS (>120 mg/dl)', (0, 1))
    resting_ecg = st.sidebar.selectbox('RestingECG', ('Normal', 'ST', 'LVH'))
    max_hr = st.sidebar.slider('MaxHR (bpm)', VALIDATION_RANGES['MaxHR'][0], VALIDATION_RANGES['MaxHR'][1], 138)
    exercise_angina = st.sidebar.selectbox('ExerciseAngina', ('N', 'Y'))
    oldpeak = st.sidebar.slider('Oldpeak', float(VALIDATION_RANGES['Oldpeak'][0]), float(VALIDATION_RANGES['Oldpeak'][1]), 0.6, step=0.1)
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
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input Features')
st.write(input_df)

if st.button('Predict Heart Disease'):
    try:
        # --- Preprocessing ---
        processed_input = input_df.copy()

        # Apply one-hot encoding for categorical features
        processed_input = pd.get_dummies(processed_input, columns=original_categorical_cols)

        # Align columns with X_train columns for consistency
        # Ensure all columns from X_train_cols are present, fill missing with 0
        # and drop any extra columns not in X_train_cols
        processed_input = processed_input.reindex(columns=X_train_cols, fill_value=0)

        # Scale numerical features
        processed_input[original_numerical_cols] = scaler.transform(processed_input[original_numerical_cols])

        # --- Prediction ---
        prediction = final_model.predict(processed_input)
        prediction_proba = final_model.predict_proba(processed_input)[:, 1]

        st.subheader('Prediction')
        result_text = "Presence of Heart Disease" if prediction[0] == 1 else "Absence of Heart Disease"
        st.success(f"Prediction: **{result_text}**")
        st.write(f"Probability of Heart Disease: **{prediction_proba[0]:.2f}**")

        # --- SHAP Explanations ---
        st.subheader('Explanation of Prediction (SHAP Values)')

        # Create a SHAP explainer for the RandomForestClassifier, outputting probabilities
        explainer = shap.TreeExplainer(final_model, model_output='probability')

        # Generate SHAP values for the processed input instance
        shap_values = explainer.shap_values(processed_input)

        # With model_output='probability', shap_values is a single array (for the positive class probability)
        # and expected_value is a scalar.
        # So, we don't need to index `shap_values[1]` or `explainer.expected_value[1]`
        shap_values_for_positive_class = shap_values[0] # For a single instance, it's the first (and only) row
        expected_value_for_positive_class = explainer.expected_value # It's already a scalar

        # SHAP Force Plot
        st.write("**How individual features contribute to the prediction:**")

        # Convert processed_input to a numpy array for SHAP plots if needed
        processed_input_array = processed_input.values

        # Force plot (usually works better in notebooks or with specific streamlit components)
        st.write("*(Force plot might not render perfectly in all Streamlit environments directly. Visualizing key contributors below.)*")

        # Waterfall Plot
        st.write("**Waterfall plot showing contribution of each feature:**")
        fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap.Explanation(values=shap_values_for_positive_class,
                                             base_values=expected_value_for_positive_class,
                                             data=processed_input_array[0],
                                             feature_names=X_train_cols),
                            show=False)
        plt.title('SHAP Waterfall Plot for Current Prediction')
        st.pyplot(plt)

        # Bar Plot of SHAP values for clarity
        st.write("**Overall impact of features on this prediction (Bar Plot):**")
        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values_for_positive_class, processed_input, plot_type="bar", show=False, feature_names=X_train_cols)
        plt.title('SHAP Feature Importance for Current Prediction')
        plt.tight_layout()
        st.pyplot(plt)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
