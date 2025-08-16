# src/utils.py
import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle

from src.exception import CustomException

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)
    


def load_object(file_path):
    """Load a Dilled Python object from disk."""
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    


def evaluate_models(X_train, X_test, y_train, y_test, models, param):
    """
    Train and evaluate multiple classifiers with error handling.
    Parameters:
        X_train, X_test, y_train, y_test: Dataset splits
        models: dict of {model_name: model_instance}
    Returns:
        results: dict containing metrics for each successfully evaluated model
    """
    results = {}
    fitted = {}

    for name, model in models.items():
        print(f"\n=== Training & Evaluating: {name} ===")
        
        try:

            grid = param.get(name,{}) #empty dict if no params

            # GridSearchCV if param porovided
            if grid:
                gs = GridSearchCV(model, grid, cv=3, n_jobs=-1, scoring="roc_auc", refit=True)
                gs.fit(X_train, y_train)
                est = gs.best_estimator_
                print(f"[GridSearchCV] best_params for {name}: {gs.best_params_}")
            else:
                model.fit(X_train, y_train)
                est = model
            
            # Predictions
            y_pred = est.predict(X_test)

            # Probabilities or decision scores for AUC
            y_score = None
            if hasattr(est, "predict_proba"):
                y_score = est.predict_proba(X_test)[:, 1]
            elif hasattr(est, "decision_function"):
                y_score = est.decision_function(X_test)

            # Metrics
            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0, pos_label=1)
            rec  = recall_score(y_test, y_pred, zero_division=0, pos_label=1)
            f1   = f1_score(y_test, y_pred, zero_division=0, pos_label=1)
            auc  = roc_auc_score(y_test, y_score) if y_score is not None else np.nan
            cm   = confusion_matrix(y_test, y_pred)

            # Report
            print("Classification Report:")
            print(classification_report(y_test, y_pred, zero_division=0))
            print(f"Accuracy   : {acc:.4f}")
            print(f"Precision  : {prec:.4f}")
            print(f"Recall     : {rec:.4f}")
            print(f"F1 Score   : {f1:.4f}")
            if not np.isnan(auc):
                print(f"AUC-ROC    : {auc:.4f}")
            print("Confusion Matrix:\n", cm)

            # Store results
            results[name] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "auc_roc": auc,
                "confusion_matrix": cm
            }
            fitted[name] = est

        except Exception as e:
            print(f"[ERROR] {name} failed: {e}")
            results[name] = {"error": str(e)}

    return results, fitted
