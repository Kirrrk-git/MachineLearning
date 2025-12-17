import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
import streamlit.components.v1 as components
from lime.lime_tabular import LimeTabularExplainer

# ------------------------------
# Load model artifacts
# ------------------------------
@st.cache_resource
def load_model_artifacts():
    model = joblib.load("best_rf_heart_disease_model.joblib")
    scaler = joblib.load("scaler.joblib")
    X_train_cols = joblib.load("X_train_columns.joblib")
    X_train_processed = joblib.load("X_train_processed.joblib")
    return model, scaler, X_train_cols, X_train_processed

model, scaler, X_train_cols, X_train_processed = load_model_artifacts()

# ------------------------------
# Feature definitions
# ------------------------------
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'FastingBS']
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

VALIDATION_RANGES = {
    'Age': (1, 100),
    'RestingBP': (50, 250),
    'Cholesterol': (50, 700),
    'MaxHR': (50, 220),
    'Oldpeak': (-3.0, 7.0)
}

# ------------------------------
# Streamlit layout
# ------------------------------
st.set_page_config(layout="wide")
st.title("Heart Disease Prediction with Explainable AI")
st.write("Enter patient details to predict heart disease risk and see interpretability visualizations (SHAP & LIME).")

# ------------------------------
# Sidebar inputs
# ------------------------------
st.sidebar.header("Patient Input Features")

def user_input_features():
    data = {
        'Age': st.sidebar.slider('Age', *VALIDATION_RANGES['Age'], 54),
        'Sex': st.sidebar.selectbox('Sex', ('M', 'F')),
        'ChestPainType': st.sidebar.selectbox('Chest Pain Type', ('ATA', 'NAP', 'ASY', 'TA')),
        'RestingBP': st.sidebar.slider('Resting Blood Pressure (mm Hg)', *VALIDATION_RANGES['RestingBP'], 130),
        'Cholesterol': st.sidebar.slider('Cholesterol (mg/dl)', *VALIDATION_RANGES['Cholesterol'], 223),
        'FastingBS': st.sidebar.selectbox('Fasting Blood Sugar >120 mg/dl', (0, 1)),
        'RestingECG': st.sidebar.selectbox('Resting ECG', ('Normal', 'ST', 'LVH')),
        'MaxHR': st.sidebar.slider('Maximum Heart Rate', *VALIDATION_RANGES['MaxHR'], 138),
        'ExerciseAngina': st.sidebar.selectbox('Exercise-Induced Angina', ('N', 'Y')),
        'Oldpeak': st.sidebar.slider('Oldpeak (ST Depression)', *VALIDATION_RANGES['Oldpeak'], 0.6, step=0.1),
        'ST_Slope': st.sidebar.selectbox('ST Slope', ('Up', 'Flat', 'Down'))
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
st.subheader("User Input Summary")
st.dataframe(input_df)

# ------------------------------
# Prediction and explanations
# ------------------------------
if st.button("Predict and Explain"):

    # ---------- Preprocessing ----------
    processed_input = pd.get_dummies(input_df, columns=categorical_cols)
    processed_input = processed_input.reindex(columns=X_train_cols, fill_value=0)
    processed_input[numerical_cols] = scaler.transform(processed_input[numerical_cols])

    # ---------- Prediction ----------
    prediction = model.predict(processed_input)[0]
    probability = model.predict_proba(processed_input)[0, 1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.warning("Prediction: **Presence of Heart Disease**")
    else:
        st.success("Prediction: **Absence of Heart Disease**")
    st.write(f"Predicted Probability of Heart Disease: **{probability:.2f}**")

    # ---------- Probability Bar Chart ----------
    st.subheader("Prediction Probability")
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.barh(["Heart Disease Risk"], [probability], color='orange' if prediction==1 else 'green')
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    st.pyplot(fig)
    st.markdown(
        f"**Interpretation:** The model predicts a **{probability:.0%} probability** of heart disease. "
        f"{'This suggests a higher cardiovascular risk.' if prediction==1 else 'This suggests a lower likelihood of heart disease.'}"
    )

    # ---------- SHAP Waterfall ----------
    st.subheader("SHAP Waterfall Explanation")
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap(processed_input)
    fig_shap, ax_shap = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0, :, 1], show=False)
    st.pyplot(fig_shap)

    shap_features_idx = np.argsort(np.abs(shap_values.values[0, :, 1]))[-3:][::-1]
    shap_features = [X_train_cols[i] for i in shap_features_idx]
    st.markdown(
        f"**SHAP Interpretation:** The most influential features were: **{', '.join(shap_features)}**. "
        "Red bars increase predicted risk, blue bars decrease it."
    )

    # ---------- LIME Explanation ----------
    st.subheader("LIME Local Explanation")
    def predict_proba_wrapper(data):
        df = pd.DataFrame(data, columns=X_train_cols)
        return model.predict_proba(df)

    explainer_lime = LimeTabularExplainer(
        training_data=X_train_processed.values,  # ✅ NumPy array
        feature_names=X_train_cols,
        class_names=['No Disease', 'Heart Disease'],
        mode='classification'
    )

    lime_exp = explainer_lime.explain_instance(
        data_row=processed_input.iloc[0].values,
        predict_fn=predict_proba_wrapper,
        num_features=5
    )

    # Wrap the HTML in a white background div to fix dark mode issues
    lime_html = f"""
    <div style="background-color:white; padding:10px; border-radius:5px;">
    {lime_exp.as_html()}
    </div>
    """

    components.html(
        lime_exp.as_html(),
        height=350,
        scrolling=True
    )

    lime_features = [f[0] for f in lime_exp.as_list()]
    st.markdown(
        f"**LIME Interpretation:** The strongest contributing features were: **{', '.join(lime_features)}**. "
        f"These features locally influenced the prediction toward **{'heart disease' if prediction==1 else 'no heart disease'}**."
    )
