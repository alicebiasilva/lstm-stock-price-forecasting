import torch
import joblib
import mlflow
import sqlite3
import pandas as pd
import numpy as np

from pathlib import Path
from src.models.lstm_model import LSTM

# =========================
# CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models/lstm_ohlcv.pt"
SCALER_PATH = BASE_DIR / "models/scaler.pkl"
DB_PATH = BASE_DIR / "data/refined/refined.db"

FEATURE_COLUMNS = ["open", "high", "low", "close", "volume"]
TARGET_INDEX = FEATURE_COLUMNS.index("close")

SEQUENCE_LENGTH = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# MLFLOW CONFIG
# =========================

MLFLOW_DIR = BASE_DIR / "mlflow"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR}/mlflow.db")
mlflow.set_experiment("lstm_stock_forecasting_ohlcv")

# =========================
# LOAD DATA
# =========================

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM prices", conn)
    conn.close()

    df.columns = df.columns.str.lower().str.strip()
    df = df.sort_values("date")

    return df

# =========================
# LOAD MODEL + SCALER
# =========================

def load_model_and_scaler():
    scaler = joblib.load(SCALER_PATH)

    model = LSTM(input_size=5).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    return model, scaler

# =========================
# INFERENCE
# =========================

def run_inference():

    print("\nIniciando inferência...")

    model, scaler = load_model_and_scaler()

    df = load_data()
    values = df[FEATURE_COLUMNS].values

    scaled = scaler.transform(values)

    last_seq = scaled[-SEQUENCE_LENGTH:]
    X = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)

    with mlflow.start_run(run_name="inference_run"):

        print("Rodando previsão...")

        with torch.no_grad():
            pred = model(X).cpu().numpy()

        # inversão correta
        dummy = np.zeros((pred.shape[0], 5))
        dummy[:, TARGET_INDEX] = pred[:, 0]

        pred_real = scaler.inverse_transform(dummy)[:, TARGET_INDEX]
        predicted_value = float(pred_real[0])

        print(f"\nPrevisão próxima: {predicted_value:.4f}")

        mlflow.log_metric("predicted_price", predicted_value)

    return predicted_value

# =========================
# RUN
# =========================

if __name__ == "__main__":
    run_inference()