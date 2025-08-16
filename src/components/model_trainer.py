# src/components/model_trainer.py

import os
import sys
from dataclasses import dataclass
import numpy as np

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models  # your utils, course-style API

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Course-style: separate `models` and `params`, hand both to evaluate_models(..., param=params).
        Hyperparameter tuning depth is intentionally small; you said you'll do full HPO later.
        """
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # ---------------------------
            # Models (sane defaults)
            # ---------------------------
            models = {
                "LogisticRegression": LogisticRegression(
                    solver="lbfgs",
                    penalty="l2",
                    class_weight="balanced",
                    max_iter=2000,
                    n_jobs=-1,
                ),
                "RandomForestClassifier": RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=42,
                ),
                "GradientBoostingClassifier": GradientBoostingClassifier(
                    random_state=42
                ),
                "AdaBoostClassifier": AdaBoostClassifier(
                    random_state=42
                ),
                "SVC": SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    probability=False,   # AUC via decision_function in your evaluator
                    random_state=42,
                ),
                "KNN": KNeighborsClassifier(
                    n_neighbors=5,
                    n_jobs=-1
                ),
                "DecisionTreeClassifier": DecisionTreeClassifier(
                    class_weight="balanced",
                    random_state=42
                ),
                "GaussianNB": GaussianNB(),
                "MLPClassifier": MLPClassifier(
                    hidden_layer_sizes=(128,),
                    max_iter=500,
                    random_state=42
                ),
                "XGBClassifier": XGBClassifier(
                    eval_metric="logloss",
                    tree_method="hist",
                    n_estimators=300,
                    random_state=42
                ),
            }

            # ---------------------------
            # Param grids 
            # Keys must match the model names above.
            # ---------------------------
            params = {
                "LogisticRegression": {
                    "C": [0.1, 1.0, 10.0]
                },
                "RandomForestClassifier": {
                    "n_estimators": [100, 200, 400],
                    "max_depth": [None, 10, 20]
                },
                "GradientBoostingClassifier": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [2, 3]
                },
                "AdaBoostClassifier": {
                    "n_estimators": [100, 300],
                    "learning_rate": [0.1, 1.0]
                },
                "SVC": {
                    "C": [0.5, 1.0, 2.0],
                    "gamma": ["scale", "auto"],
                    "kernel": ["rbf"]
                },
                "KNN": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"]
                },
                "DecisionTreeClassifier": {
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 10]
                },
                "GaussianNB": {
                    # tiny smoothing sweep; cheap
                    "var_smoothing": [1e-9, 1e-8, 1e-7]
                },
                "MLPClassifier": {
                    "hidden_layer_sizes": [(64,), (128,)],
                    "alpha": [1e-4, 1e-3]
                },
                "XGBClassifier": {
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5],
                    "n_estimators": [200, 300]
                },
            }

            logging.info("Evaluating models with course-style evaluator")
            model_report, fitted = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test,   y_test=y_test,
                models=models, 
                param=params
            )
            if not model_report:
                raise CustomException("evaluate_models returned empty report", sys)
            
            # Rank by AUC first, f1 as tiebreaker (handles Nan Safely)
            PRIMARY, FALLBACK = "auc_roc", "f1_score"

            def score_tuple(m:dict):
                auc = np.nan_to_num(m.get(PRIMARY, np.nan), nan=-1.0)
                f1 = np.nan_to_num(m.get(FALLBACK, np.nan), nan=-1.0)
                return (auc, f1)

            
            best_model_name = max(model_report.keys(), key=lambda k: score_tuple(model_report[k]))
            best_metric = model_report[best_model_name]

            # Use AUC if present else F1 for theshold/logging
            best_model_score = (
                best_metric[PRIMARY]
                if not np.isnan(best_metric.get(PRIMARY, np.nan))
                else best_metric[FALLBACK]
            )

            # Minimal sanity check; adjust to your tolerance
            if best_model_score < 0.6:
                raise CustomException(f"No best model found (best={best_model_name}, score={best_model_score:.4f})", sys)

            logging.info(f"Best model: {best_model_name} | score: {best_model_score:.4f}")


            best_model = fitted[best_model_name]
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Saved best model to {self.model_trainer_config.trained_model_file_path} | score: {best_model_score:.4f}")
            
            return (
                best_model_score,
                best_model_name
            )
        

        except Exception as e:
            logging.error(f"initiate_model_trainer failed: {e}")
            raise CustomException(e, sys)
