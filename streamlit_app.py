# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉", layout="centered")

st.title("Customer Churn Prediction")
st.caption("Uses your saved preprocessor and model from artifact/")

@st.cache_resource(show_spinner=False)
def load_pipeline():
    # Lazy-load once; prevents reloading on every widget change
    return PredictPipeline()

pipe = load_pipeline()

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        customerID = st.text_input("Customer ID", "Dummy")
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
        SeniorCitizen = st.selectbox("SeniorCitizen", ["No", "Yes"], index=0)
        Partner = st.selectbox("Partner", ["No", "Yes"], index=0)
        Dependents = st.selectbox("Dependents", ["No", "Yes"], index=0)
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=1000, value=12, step=1)
        PhoneService = st.selectbox("PhoneService", ["Yes", "No"], index=0)
        MultipleLines = st.selectbox("MultipleLines", ["No", "Yes", "No phone service"], index=0)
        InternetService = st.selectbox("InternetService", ["Fiber optic", "DSL", "No"], index=0)

    with col2:
        OnlineSecurity = st.selectbox("OnlineSecurity", ["No", "Yes", "No internet service"], index=0)
        OnlineBackup = st.selectbox("OnlineBackup", ["No", "Yes", "No internet service"], index=0)
        DeviceProtection = st.selectbox("DeviceProtection", ["No", "Yes", "No internet service"], index=0)
        TechSupport = st.selectbox("TechSupport", ["No", "Yes", "No internet service"], index=0)
        StreamingTV = st.selectbox("StreamingTV", ["No", "Yes", "No internet service"], index=0)
        StreamingMovies = st.selectbox("StreamingMovies", ["No", "Yes", "No internet service"], index=0)
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=0)
        PaperlessBilling = st.selectbox("PaperlessBilling", ["Yes", "No"], index=0)
        PaymentMethod = st.selectbox(
            "PaymentMethod",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            index=0,
        )
        MonthlyCharges = st.number_input("MonthlyCharges", min_value=0.0, max_value=10000.0, value=70.0, step=1.0)
        TotalCharges_raw = st.text_input("TotalCharges (leave blank to auto-fill)", "")

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        # Safe parse: let your preprocessor compute tenure * MonthlyCharges when blank
        TotalCharges = 0.0 if str(TotalCharges_raw).strip() == "" else float(TotalCharges_raw)

        data = CustomData(
            customerID=customerID,
            gender=gender,
            SeniorCitizen=1 if SeniorCitizen == "Yes" else 0,
            Partner=Partner,
            Dependents=Dependents,
            tenure=int(tenure),
            PhoneService=PhoneService,
            MultipleLines=MultipleLines,
            InternetService=InternetService,
            OnlineSecurity=OnlineSecurity,
            OnlineBackup=OnlineBackup,
            DeviceProtection=DeviceProtection,
            TechSupport=TechSupport,
            StreamingTV=StreamingTV,
            StreamingMovies=StreamingMovies,
            Contract=Contract,
            PaperlessBilling=PaperlessBilling,
            PaymentMethod=PaymentMethod,
            MonthlyCharges=float(MonthlyCharges),
            TotalCharges=TotalCharges,
        )

        df = data.get_data_as_frame()

        with st.spinner("Running model..."):
            preds, probs = pipe.predict(df)

        p = float(probs[0]) if probs is not None else None
        label = int(preds[0])

        st.subheader("Result")
        if p is not None:
            st.metric("Churn probability", f"{p:.2%}")
            st.progress(min(max(p, 0.0), 1.0))
        else:
            st.write("Model doesn’t expose probabilities; showing class only.")

        st.write(f"Predicted class: **{label}**  (1 = churn, 0 = stay)")

        with st.expander("Submitted features"):
            st.dataframe(df)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
