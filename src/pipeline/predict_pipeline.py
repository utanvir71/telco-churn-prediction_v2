# src/pipeline/predict_pipeline.py

import sys
import os
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.utils import load_object  # make sure this loads with the SAME lib you used to save (dill<->dill)

ART_DIR = "artifact"
MODEL_FP = os.path.join(ART_DIR, "model.pkl")
PREPROC_FP = os.path.join(ART_DIR, "preprocessor.pkl")


class PredictPipeline:
    def __init__(self):
        try:
            self.model = load_object(MODEL_FP)
            self.preprocessor = load_object(PREPROC_FP)
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame):
        """
        features: DataFrame with the raw feature columns used in training (no 'Churn')
        returns: (preds, probs) where probs may be None if estimator has no scores
        """
        try:
            X_t = self.preprocessor.transform(features)

            preds = self.model.predict(X_t)

            probs = None
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X_t)[:, 1]
            elif hasattr(self.model, "decision_function"):
                scores = self.model.decision_function(X_t)
                probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

            return preds, probs
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    One-row input matching training schema (excluding 'Churn' and 'customerID').
    """
    def __init__(
        self,
        customerID: str,
        gender: str,
        SeniorCitizen: int,
        Partner: str,
        Dependents: str,
        tenure: int,
        PhoneService: str,
        MultipleLines: str,
        InternetService: str,
        OnlineSecurity: str,
        OnlineBackup: str,
        DeviceProtection: str,
        TechSupport: str,
        StreamingTV: str,
        StreamingMovies: str,
        Contract: str,
        PaperlessBilling: str,
        PaymentMethod: str,
        MonthlyCharges: float,
        TotalCharges: float,
    ):
        self.customerID = customerID
        self.gender = gender
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.tenure = tenure
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges

    def get_data_as_frame(self) -> pd.DataFrame:
        try:
            row = {
                "customerID": self.customerID,
                "gender": self.gender,
                "SeniorCitizen": self.SeniorCitizen,
                "Partner": self.Partner,
                "Dependents": self.Dependents,
                "tenure": self.tenure,
                "PhoneService": self.PhoneService,
                "MultipleLines": self.MultipleLines,
                "InternetService": self.InternetService,
                "OnlineSecurity": self.OnlineSecurity,
                "OnlineBackup": self.OnlineBackup,
                "DeviceProtection": self.DeviceProtection,
                "TechSupport": self.TechSupport,
                "StreamingTV": self.StreamingTV,
                "StreamingMovies": self.StreamingMovies,
                "Contract": self.Contract,
                "PaperlessBilling": self.PaperlessBilling,
                "PaymentMethod": self.PaymentMethod,
                "MonthlyCharges": self.MonthlyCharges,
                "TotalCharges": self.TotalCharges,
            }
            return pd.DataFrame([row])
        except Exception as e:
            raise CustomException(e, sys)


# Quick self-test
if __name__ == "__main__":
    try:
        sample = CustomData(
            customerID="Dummy",
            gender="Male",
            SeniorCitizen=0,
            Partner="Yes",
            Dependents="No",
            tenure=12,
            PhoneService="Yes",
            MultipleLines="No",
            InternetService="Fiber optic",
            OnlineSecurity="No",
            OnlineBackup="Yes",
            DeviceProtection="No",
            TechSupport="No",
            StreamingTV="Yes",
            StreamingMovies="No",
            Contract="Month-to-month",
            PaperlessBilling="Yes",
            PaymentMethod="Electronic check",
            MonthlyCharges=70.35,
            TotalCharges=845.50,
        ).get_data_as_frame()

        pipe = PredictPipeline()
        yhat, p = pipe.predict(sample)
        print("Pred:", int(yhat[0]), "Prob:", None if p is None else float(np.round(p[0], 4)))
    except Exception as err:
        raise CustomException(err, sys)
