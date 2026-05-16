import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import mlflow
import mlflow.pytorch

import sqlite3
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from src.models.lstm_model import LSTM
from pathlib import Path

# =========================
# CONFIG
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sequence_length = 60
batch_size = 64
num_epochs = 50
learning_rate = 0.001

FEATURE_COLUMNS = ["open", "high", "low", "close", "volume"]
TARGET_INDEX = FEATURE_COLUMNS.index("close")

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "data/refined/refined.db"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MLFLOW_DIR = BASE_DIR / "mlflow"
MLFLOW_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR}/mlflow.db")
mlflow.set_experiment("lstm_stock_forecasting_ohlcv")

# Early stopping config
PATIENCE = 5
best_val_loss = float("inf")
early_stop_counter = 0

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
# SEQUENCES
# =========================

def create_sequences(data, seq_len):
    X, y = [], []

    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, TARGET_INDEX])

    return np.array(X), np.array(y)

# =========================
# PREP DATA
# =========================

def prepare_data():

    df = load_data()

    features = df[FEATURE_COLUMNS].values

    train_size = int(len(features) * 0.7)
    val_size = int(len(features) * 0.15)

    train = features[:train_size]
    val = features[train_size:train_size+val_size]
    test = features[train_size+val_size:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)

    X_train, y_train = create_sequences(train_scaled, sequence_length)
    X_val, y_val = create_sequences(val_scaled, sequence_length)
    X_test, y_test = create_sequences(test_scaled, sequence_length)

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
        scaler
    )

# =========================
# TRAIN
# =========================

def train():

    global best_val_loss, early_stop_counter

    (X_train, y_train,
     X_val, y_val,
     X_test, y_test,
     scaler) = prepare_data()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    model = LSTM(input_size=5).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scaler_path = MODEL_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    best_model_path = MODEL_DIR / "lstm_ohlcv.pt"

    with mlflow.start_run():

        mlflow.log_params({
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "features": "OHLCV",
            "early_stopping_patience": PATIENCE
        })

        for epoch in range(num_epochs):

            # =========================
            # TRAIN
            # =========================

            model.train()
            total_loss = 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                preds = model(X_batch).squeeze()
                loss = criterion(preds, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # =========================
            # VALIDATION
            # =========================

            model.eval()
            with torch.no_grad():
                val_preds = model(X_val.to(device)).cpu().numpy().flatten()
                val_true = y_val.numpy()

            val_loss = mean_squared_error(val_true, val_preds)

            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.6f} - Val Loss: {val_loss:.6f}")

            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            # =========================
            # EARLY STOPPING
            # =========================

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0

                # salva melhor modelo
                torch.save(model.state_dict(), best_model_path)

            else:
                early_stop_counter += 1

            if early_stop_counter >= PATIENCE:
                print("\nEarly stopping ativado")
                break

        # =========================
        # LOAD BEST MODEL
        # =========================

        model.load_state_dict(torch.load(best_model_path))

        # =========================
        # TEST
        # =========================

        model.eval()
        with torch.no_grad():
            test_preds = model(X_test.to(device)).cpu().numpy().flatten()
            test_true = y_test.numpy()

        test_rmse = mean_squared_error(test_true, test_preds) ** 0.5

        print(f"\nTest RMSE: {test_rmse:.4f}")

        mlflow.log_metric("test_rmse", test_rmse)

        mlflow.log_artifact(str(best_model_path))
        mlflow.log_artifact(str(scaler_path))

        print("\nTreinamento finalizado.")

if __name__ == "__main__":
    train()