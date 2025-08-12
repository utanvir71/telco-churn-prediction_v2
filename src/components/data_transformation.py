# src/components/data_transformation.py
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object  # dill-based saver

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifact"


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = str(ARTIFACT_DIR / "preprocessor.pkl")


def _fix_total_charges_fn(X: pd.DataFrame) -> pd.DataFrame:
    """
    Pure function for FunctionTransformer:
    - Coerce TotalCharges to numeric (errors='coerce')
    - Where TotalCharges is NaN, set to tenure * MonthlyCharges
    """
    if not isinstance(X, pd.DataFrame):
        # Expect DataFrame throughout; if someone passes ndarray, convert if possible
        X = pd.DataFrame(X)
    if "TotalCharges" in X.columns:
        X = X.copy()
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        mask = X["TotalCharges"].isna()
        # If tenure/MonthlyCharges are NaN, this will still be NaN, which the numeric imputer will handle
        if "tenure" in X.columns and "MonthlyCharges" in X.columns:
            X.loc[mask, "TotalCharges"] = (
                pd.to_numeric(X.loc[mask, "tenure"], errors="coerce")
                * pd.to_numeric(X.loc[mask, "MonthlyCharges"], errors="coerce")
            )
    return X


def _make_ohe():
    """Return a OneHotEncoder compatible with your sklearn version."""
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # sklearn <= 1.1
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self) -> Pipeline:
        try:
            numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
            categorical_cols = [
                "gender", "Partner", "Dependents", "PhoneService",
                "MultipleLines", "InternetService", "OnlineSecurity",
                "OnlineBackup", "DeviceProtection", "TechSupport",
                "StreamingTV", "StreamingMovies", "Contract",
                "PaperlessBilling", "PaymentMethod",
            ]

            # Column-wise transformers
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _make_ohe()),
            ])

            column_wise = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numeric_cols),
                    ("cat", cat_pipeline, categorical_cols),
                ],
                remainder="drop",
            )

            # Full pipeline: first fix TotalCharges, then column-wise transform
            full_preprocessor = Pipeline([
                ("fix_total_charges", FunctionTransformer(_fix_total_charges_fn, feature_names_out="one-to-one")),
                ("columns", column_wise),
            ])

            logging.info("Full preprocessor (with TotalCharges fix) created.")
            return full_preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)

            logging.info("Loaded train/test data.")

            target_col = "Churn"
            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col].map({"Yes": 1, "No": 0}).astype(int)

            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col].map({"Yes": 1, "No": 0}).astype(int)

            preprocessor = self.get_data_transformer_object()

            # This will automatically fix TotalCharges via FunctionTransformer
            X_train_t = preprocessor.fit_transform(X_train)
            X_test_t  = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_t, y_train.to_numpy()]
            test_arr  = np.c_[X_test_t,  y_test.to_numpy()]

            # Save the **fitted** full preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor,
            )
            logging.info(f"Saved preprocessor to {self.data_transformation_config.preprocessor_obj_file_path}")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.exception("Data transformation failed.")
            raise CustomException(e, sys)
