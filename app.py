import streamlit as st
import pandas as pd
import numpy as np
import joblib
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
st.write("Enter patient details to predict the likelihood of heart disease.")

# --- Input Widgets ---
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

        # Ensure boolean columns are converted to int for compatibility
        for col in processed_input.columns:
            if processed_input[col].dtype == 'bool':
                processed_input[col] = processed_input[col].astype(int)

        # Scale numerical features
        processed_input[original_numerical_cols] = scaler.transform(processed_input[original_numerical_cols])

        # --- Prediction ---
        prediction = final_model.predict(processed_input)
        prediction_proba = final_model.predict_proba(processed_input)[:, 1]

        st.subheader('Prediction Result')
        result_text = "Presence of Heart Disease" if prediction[0] == 1 else "Absence of Heart Disease"
        
        if prediction[0] == 1:
            st.warning(f"Prediction: **{result_text}**") # Orange for heart disease
        else:
            st.success(f"Prediction: **{result_text}**") # Green for no heart disease
            
        st.write(f"Probability of Heart Disease: **{prediction_proba[0]:.2f}**")

        # --- Visual Illustration of Prediction Probability ---
        st.subheader('Prediction Probability Visual')
        fig, ax = plt.subplots(figsize=(8, 2))
        colors = ['orange' if prediction[0] == 1 else 'green']
        ax.barh(['Probability'], [prediction_proba[0]], color=colors)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel('Probability')
        ax.set_title('Predicted Probability of Heart Disease')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# --- Dataset Attributes Explained ---
st.header("Dataset Attributes Explained")
st.markdown("""
Here's a breakdown of each attribute in the heart disease dataset, explaining its purpose, how it's measured, and its relevance:

*   **Age:**
    *   **Purpose/Role:** Represents the patient's age. Age is a significant demographic risk factor for various diseases, including heart disease.
    *   **Measurement:** Recorded in years.

*   **Sex:**
    *   **Purpose/Role:** Indicates the biological sex of the patient. Sex can influence disease prevalence, symptoms, and risk factors.
    *   **Measurement:** Categorical, with 'M' for Male and 'F' for Female.

*   **ChestPainType:**
    *   **Purpose/Role:** Describes the type of chest pain experienced by the patient. Different types of chest pain have varying implications for heart health.
    *   **Measurement:** Categorical:
        *   **TA:** Typical Angina (chest pain due to reduced blood flow to the heart).
        *   **ATA:** Atypical Angina (chest pain not typical of angina but still potentially cardiac).
        *   **NAP:** Non-Anginal Pain (chest pain not related to the heart).
        *   **ASY:** Asymptomatic (no chest pain symptoms, but may still have underlying heart disease).

*   **RestingBP (Resting Blood Pressure):**
    *   **Purpose/Role:** Measures the pressure of blood against artery walls when the heart is at rest. High resting blood pressure (hypertension) is a major risk factor for heart disease.
    *   **Measurement:** Numeric value, measured in millimeters of mercury (mm Hg).

*   **Cholesterol (Serum Cholesterol):**
    *   **Purpose/Role:** Refers to the level of cholesterol in the blood. High levels of certain types of cholesterol can lead to plaque buildup in arteries, increasing heart disease risk.
    *   **Measurement:** Numeric value, measured in milligrams per deciliter (mg/dl).

*   **FastingBS (Fasting Blood Sugar):**
    *   **Purpose/Role:** Indicates the patient's blood glucose level after a period of fasting. High fasting blood sugar is a marker for diabetes, which is a significant risk factor for heart disease.
    *   **Measurement:** Binary categorical:
        *   **1:** if Fasting Blood Sugar is greater than 120 mg/dl (indicating high sugar).
        *   **0:** otherwise (normal or low sugar).

*   **RestingECG (Resting Electrocardiogram Results):**
    *   **Purpose/Role:** Results from an electrocardiogram (ECG) performed while the patient is at rest. ECGs detect electrical activity of the heart and can reveal abnormalities that suggest heart conditions.
    *   **Measurement:** Categorical:
        *   **Normal:** Normal heart electrical activity.
        *   **ST:** Indicates ST-T wave abnormality (changes in the ECG waveform associated with myocardial ischemia).
        *   **LVH:** Shows probable or definite left ventricular hypertrophy (enlargement of the heart's main pumping chamber).

*   **MaxHR (Maximum Heart Rate Achieved):**
    *   **Purpose/Role:** The highest heart rate reached during an exercise stress test. A lower maximum heart rate for a given age can sometimes be an indicator of heart issues.
    *   **Measurement:** Numeric value, maximum heart rate is about 220 minus current age.

*   **ExerciseAngina (Exercise-Induced Angina):**
    *   **Purpose/Role:** Determines if chest pain (angina) is triggered by physical exertion. Angina that occurs with exercise is a strong symptom of coronary artery disease.
    *   **Measurement:** Binary categorical: 'Y' for Yes (angina induced) and 'N' for No (no angina induced).

*   **Oldpeak (ST Depression Induced by Exercise Relative to Rest):**
    *   **Purpose/Role:** A measure derived from an exercise ECG, representing the extent of ST segment depression during exercise compared to rest. Greater depression can indicate myocardial ischemia (reduced blood flow to the heart muscle).
    *   **Measurement:** Numeric value, measured in depression. Can be positive or negative.

*   **ST_Slope (Slope of the Peak Exercise ST Segment):**
    *   **Purpose/Role:** Describes the direction of the ST segment's slope on an exercise ECG. This is a key indicator of myocardial ischemia.
    *   **Measurement:** Categorical:
        *   **Up:** Upsloping (generally considered normal or less indicative of disease).
        *   **Flat:** Flat (suggests ischemia).
        *   **Down:** Downsloping (strongest indicator of ischemia).

*   **HeartDisease (Output Class):**
    *   **Purpose/Role:** The target variable indicating the presence or absence of heart disease in the patient.
    *   **Measurement:** Binary categorical:
        *   **1:** Presence of heart disease.
        *   **0:** Absence of heart disease (Normal).

These attributes collectively provide a comprehensive profile of a patient's cardiac health, allowing the machine learning model to identify patterns associated with heart disease.
""")
