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



# --- Batch predictions (CSV) ---
st.markdown("## 📥 Batch predictions (CSV)")
csv = st.file_uploader("Upload a CSV with the same columns as the form", type=["csv"])

if csv is not None:
    try:
        batch_df = pd.read_csv(csv)
        with st.spinner("Scoring…"):
            preds, probs = pipe.predict(batch_df)

        out = batch_df.copy()
        out["churn_pred"] = preds.astype(int)
        if probs is not None:
            out["churn_prob"] = probs.astype(float)

        st.success(f"Scored {len(out):,} rows")
        st.dataframe(out.head(20), use_container_width=True)

        # download
        st.download_button(
            "Download predictions CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")



        # --- Risk distribution ---
if csv is not None and "churn_prob" in out.columns:
    st.markdown("## 📊 Churn risk distribution")
    # simple bins
    counts, bins = np.histogram(out["churn_prob"], bins=[0,0.2,0.4,0.6,0.8,1.0])
    dist = pd.DataFrame({"bin": [f"{bins[i]:.1f}–{bins[i+1]:.1f}" for i in range(len(bins)-1)],
                         "count": counts})
    st.bar_chart(dist.set_index("bin"))


    # --- Feature importance (Permutation Importance) ---
from sklearn.inspection import permutation_importance

st.markdown("## 🧠 Feature importance (permutation)")
with st.expander("Compute on a small sample (fast)"):
    try:
        # grab a tiny sample from the last uploaded CSV if available,
        # otherwise build one row from the form to avoid empty data
        if csv is not None and len(batch_df) > 50:
            sample = batch_df.sample(min(200, len(batch_df)), random_state=42)
        else:
            sample = df.copy()  # the single-row form dataframe

        # use the same preprocessor + model inside your pipeline
        pre = pipe.preprocessor   # make sure PredictPipeline exposes this
        clf = pipe.model          # and this

        X = pre.transform(sample)             # numpy array after transforms
        y_pred = clf.predict(X)               # baseline predictions

        # wrap a callable for permutation_importance
        def predict_fn(Xnum):
            return clf.predict_proba(Xnum)[:,1] if hasattr(clf, "predict_proba") else clf.decision_function(Xnum)

        r = permutation_importance(
            clf, X, y_pred, n_repeats=5, random_state=42, scoring=None
        )

        # map back to feature names (this works if your preprocessor has get_feature_names_out)
        try:
            feat_names = pre.get_feature_names_out()
        except Exception:
            feat_names = [f"f{i}" for i in range(X.shape[1])]

        imp_df = pd.DataFrame({"feature": feat_names, "importance": r.importances_mean})
        imp_df = imp_df.sort_values("importance", ascending=False).head(30)

        st.dataframe(imp_df, use_container_width=True)
        st.bar_chart(imp_df.set_index("feature"))
    except AttributeError:
        st.info("Your PredictPipeline must expose `preprocessor` and `model` to compute importances. Add them and try again.")
    except Exception as e:
        st.error(f"Feature importance failed: {e}")