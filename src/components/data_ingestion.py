# src/components/data_ingestion.py
import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    artifact_dir: str = os.path.join("artifact")
    raw_data_path: str = os.path.join(artifact_dir, "data.csv")
    train_data_path: str = os.path.join(artifact_dir, "train.csv")
    test_data_path: str  = os.path.join(artifact_dir, "test.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Reading dataset")
            df = pd.read_csv("notebook/data/telco_churn.csv")

            logging.info("Creating artifact directory if missing")
            os.makedirs(self.config.artifact_dir, exist_ok=True)

            logging.info("Saving raw data")
            df.to_csv(self.config.raw_data_path, index=False)

            logging.info("Train/test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            logging.info("Saving train and test CSVs")
            train_set.to_csv(self.config.train_data_path, index=False)
            test_set.to_csv(self.config.test_data_path, index=False)

            logging.info("Data ingestion completed")
            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)