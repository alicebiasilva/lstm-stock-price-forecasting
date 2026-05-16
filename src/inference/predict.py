import torch
import numpy as np
import joblib
import mlflow
import sqlite3
import pandas as pd

from pathlib import Path
from src.models.lstm_model import LSTM

# =========================
# CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models/lstm_model.pt"
SCALER_PATH = BASE_DIR / "models/scaler.pkl"
DB_PATH = BASE_DIR / "data/refined/refined.db"

sequence_length = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD DATA
# =========================

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM prices", conn)
    conn.close()

    df = df.sort_values("date")
    return df

# =========================
# CREATE SEQUENCE
# =========================

def create_last_sequence(data, seq_len):
    return data[-seq_len:]

# =========================
# INFERENCE
# =========================

def run_inference():

    print("\nCarregando modelo e scaler...")

    scaler = joblib.load(SCALER_PATH)

    model = LSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    df = load_data()
    values = df["close"].values.reshape(-1, 1)

    scaled = scaler.transform(values)

    last_seq = create_last_sequence(scaled, sequence_length)
    X = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)

    with mlflow.start_run(run_name="inference_run"):

        print("\nRodando inferência...")

        with torch.no_grad():
            pred = model(X).cpu().numpy()

        pred_real = scaler.inverse_transform(pred)

        print(f"\nPrevisão próxima: {pred_real.flatten()[0]:.4f}")

        # =========================
        # LOG NO MLFLOW
        # =========================

        mlflow.log_param("model", "LSTM")
        mlflow.log_param("sequence_length", sequence_length)

        mlflow.log_metric("predicted_price", float(pred_real[0][0]))

        mlflow.set_tag("type", "inference")

        print("\nInferência registrada no MLflow.")

# =========================
# RUN
# =========================

if __name__ == "__main__":
    run_inference()