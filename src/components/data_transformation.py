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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object  # dill-based saver

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifact"

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = str(ARTIFACT_DIR / "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    @staticmethod
    def _fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
        if "TotalCharges" in df.columns:
            df = df.copy()
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            mask = df["TotalCharges"].isna()
            df.loc[mask, "TotalCharges"] = (
                df.loc[mask, "tenure"] * df.loc[mask, "MonthlyCharges"]
            )
        return df

    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
            categorical_cols = [
                "gender", "Partner", "Dependents", "PhoneService",
                "MultipleLines", "InternetService", "OnlineSecurity",
                "OnlineBackup", "DeviceProtection", "TechSupport",
                "StreamingTV", "StreamingMovies", "Contract",
                "PaperlessBilling", "PaymentMethod",
            ]

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])

            # Use sparse=False for broad sklearn compatibility
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numeric_cols),
                    ("cat", cat_pipeline, categorical_cols),
                ],
                remainder="drop",
            )
            logging.info("Preprocessor created.")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            # Ensure artifact dir exists (absolute)
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df = self._fix_total_charges(train_df)
            test_df  = self._fix_total_charges(test_df)

            logging.info("Read train/test and fixed TotalCharges.")

            target_col = "Churn"
            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col].map({"Yes": 1, "No": 0}).astype(int)

            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col].map({"Yes": 1, "No": 0}).astype(int)

            preprocessor = self.get_data_transformer_object()
            X_train_t = preprocessor.fit_transform(X_train)
            X_test_t  = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_t, y_train.to_numpy()]
            test_arr  = np.c_[X_test_t, y_test.to_numpy()]

            # Save the fitted preprocessor to an absolute path
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor,
            )
            logging.info(f"Saved preprocessor to {self.data_transformation_config.preprocessor_obj_file_path}")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            # Surface the real reason instead of silently failing
            logging.exception("Data transformation failed.")
            raise CustomException(e, sys)
