import streamlit as st
import pandas as pd
import numpy as np

# Safe ML imports
import joblib
import matplotlib.pyplot as plt

# Optional explainability imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


# -------------------------------------------------
# Load model artifacts
# -------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    model = joblib.load("best_rf_heart_disease_model.joblib")
    scaler = joblib.load("scaler.joblib")
    X_train_cols = joblib.load("X_train_columns.joblib")
    return model, scaler, X_train_cols


model, scaler, X_train_cols = load_model_artifacts()

# -------------------------------------------------
# Feature definitions
# -------------------------------------------------
categorical_cols = [
    'Sex', 'ChestPainType', 'RestingECG',
    'ExerciseAngina', 'ST_Slope', 'FastingBS'
]

numerical_cols = [
    'Age', 'RestingBP', 'Cholesterol',
    'MaxHR', 'Oldpeak'
]

VALIDATION_RANGES = {
    'Age': (1, 100),
    'RestingBP': (50, 250),
    'Cholesterol': (50, 700),
    'MaxHR': (50, 220),
    'Oldpeak': (-3.0, 7.0)
}

# -------------------------------------------------
# Streamlit Layout
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("Heart Disease Prediction with Explainable AI")

st.write(
    "This application predicts the likelihood of heart disease "
    "using a trained machine learning model."
)

# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------
st.sidebar.header("Patient Information")

def user_input_features():
    data = {
        'Age': st.sidebar.slider('Age', *VALIDATION_RANGES['Age'], 54),
        'Sex': st.sidebar.selectbox('Sex', ('M', 'F')),
        'ChestPainType': st.sidebar.selectbox('Chest Pain Type', ('ATA', 'NAP', 'ASY', 'TA')),
        'RestingBP': st.sidebar.slider('Resting BP (mm Hg)', *VALIDATION_RANGES['RestingBP'], 130),
        'Cholesterol': st.sidebar.slider('Cholesterol (mg/dl)', *VALIDATION_RANGES['Cholesterol'], 223),
        'FastingBS': st.sidebar.selectbox('Fasting Blood Sugar >120 mg/dl', (0, 1)),
        'RestingECG': st.sidebar.selectbox('Resting ECG', ('Normal', 'ST', 'LVH')),
        'MaxHR': st.sidebar.slider('Max Heart Rate', *VALIDATION_RANGES['MaxHR'], 138),
        'ExerciseAngina': st.sidebar.selectbox('Exercise-Induced Angina', ('N', 'Y')),
        'Oldpeak': st.sidebar.slider('Oldpeak', *VALIDATION_RANGES['Oldpeak'], 0.6, step=0.1),
        'ST_Slope': st.sidebar.selectbox('ST Slope', ('Up', 'Flat', 'Down'))
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader("User Input Summary")
st.dataframe(input_df)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("Predict Heart Disease"):

    processed = pd.get_dummies(input_df, columns=categorical_cols)
    processed = processed.reindex(columns=X_train_cols, fill_value=0)
    processed[numerical_cols] = scaler.transform(processed[numerical_cols])

    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0, 1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.warning("Prediction: **Presence of Heart Disease**")
    else:
        st.success("Prediction: **Absence of Heart Disease**")

    st.write(f"Probability of Heart Disease: **{probability:.2f}**")

    # -------------------------------------------------
    # Probability Visualization
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 2))
    ax.barh(["Heart Disease Risk"], [probability])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    st.pyplot(fig)

    st.markdown(
        f"""
        **Interpretation:**  
        The model estimates a **{probability:.0%} probability** of heart disease.
        {'This suggests elevated cardiovascular risk.' if prediction == 1 else
         'This suggests a lower likelihood of heart disease.'}
        """
    )

    # -------------------------------------------------
    # SHAP Explanation (if available)
    # -------------------------------------------------
    if SHAP_AVAILABLE:
        st.subheader("SHAP Explanation")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(processed)

        fig_shap = plt.figure()
        shap.plots.waterfall(shap_values[0, :, 1], show=False)
        st.pyplot(fig_shap)

    else:
        st.info("SHAP is not available in the current environment.")

    # -------------------------------------------------
    # LIME Explanation (if available)
    # -------------------------------------------------
    if LIME_AVAILABLE:
        st.subheader("LIME Explanation")

        explainer_lime = LimeTabularExplainer(
            training_data=processed.values,
            feature_names=X_train_cols,
            class_names=['No Disease', 'Heart Disease'],
            mode='classification'
        )

        lime_exp = explainer_lime.explain_instance(
            processed.iloc[0].values,
            model.predict_proba,
            num_features=5
        )

        fig_lime = lime_exp.as_pyplot_figure()
        st.pyplot(fig_lime)

    else:
        st.info("LIME is not available in the current environment.")
