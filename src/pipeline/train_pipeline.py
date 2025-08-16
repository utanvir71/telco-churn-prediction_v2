# src/pipeline/train_pipline.py
import sys
import os
import subprocess
import webbrowser
from pathlib import Path

from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[2]      # repo root
ARTIFACT_DIR = PROJECT_ROOT / "artifact"
STREAMLIT_FILE = PROJECT_ROOT / "streamlit_app.py"

def train():
    logging.info("Training pipeline started...")
    # 1) Ingest
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # 2) Transform
    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)

    # 3) Train model
    trainer = ModelTrainer()
    score, name = trainer.initiate_model_trainer(train_arr, test_arr)
    logging.info(f"Training finished. Best model: {name} | score: {score:.4f}")

    # Sanity: ensure artifacts exist
    need = [ARTIFACT_DIR / "preprocessor.pkl", ARTIFACT_DIR / "model.pkl"]
    for fp in need:
        if not fp.exists():
            raise CustomException(f"Expected artifact not found: {fp}", sys)

def launch_streamlit(port: int = 8501, address: str = "0.0.0.0"):
    if not STREAMLIT_FILE.exists():
        raise CustomException(f"Streamlit file not found at {STREAMLIT_FILE}", sys)

    # Use the same Python that ran training (so the env matches)
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(STREAMLIT_FILE),
        "--server.port", str(port),
        "--server.address", address,
    ]

    logging.info(f"Starting Streamlit with: {' '.join(cmd)}")
    # Open browser proactively (non-blocking)
    try:
        webbrowser.open_new_tab(f"http://{address if address!='0.0.0.0' else 'localhost'}:{port}")
    except Exception:
        pass

    # Hand over control to Streamlit; keep it attached to current terminal
    # If you want to return immediately, switch to subprocess.Popen(cmd)
    os.execvp(cmd[0], cmd)

if __name__ == "__main__":
    try:
        # Optional skip via env var if you ever need just-training
        if os.getenv("SKIP_STREAMLIT", "").lower() in {"1", "true", "yes"}:
            train()
            print("Training complete. SKIP_STREAMLIT set, not launching app.")
        else:
            train()
            # Change port if 8501 is busy
            port = int(os.getenv("STREAMLIT_PORT", "8501"))
            launch_streamlit(port=port, address=os.getenv("STREAMLIT_ADDR", "0.0.0.0"))
    except Exception as e:
        logging.error("Error in training pipeline")
        raise CustomException(e, sys)

