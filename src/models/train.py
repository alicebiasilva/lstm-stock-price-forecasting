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
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mlflow.models.signature import infer_signature
from src.models.lstm_model import LSTM
from pathlib import Path

# =========================
# CONFIG
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sequence_length = 20
batch_size = 64
num_epochs = 20
learning_rate = 0.001

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "data/refined/refined.db"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MLFLOW_DB = BASE_DIR / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
mlflow.set_experiment("lstm_stock_forecasting_ohlcv")

# =========================
# LOAD DATA
# =========================

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM prices", conn)
    conn.close()

    df.columns = df.columns.str.lower()
    df = df.sort_values("date")

    return df


# =========================
# SEQUENCES
# =========================

def create_sequences(data, seq_len):
    X, y = [], []

    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 3])  # close como target

    return np.array(X), np.array(y)


# =========================
# PREP DATA
# =========================

def prepare_data():

    df = load_data()

    print("\nUsando features OHLCV...")

    features = df[["open", "high", "low", "close", "volume"]].values

    # =========================
    # SPLIT TEMPORAL
    # =========================
    train_size = int(len(features) * 0.7)
    val_size = int(len(features) * 0.15)

    train = features[:train_size]
    val = features[train_size:train_size+val_size]
    test = features[train_size+val_size:]

    # =========================
    # SCALER (SÓ TREINO)
    # =========================
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)

    # =========================
    # SEQUENCES
    # =========================
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
        scaler,
        df
    )


# =========================
# METRICS
# =========================

def directional_accuracy(y_true, y_pred):
    return np.mean(
        np.sign(y_pred[1:] - y_pred[:-1]) ==
        np.sign(y_true[1:] - y_true[:-1])
    )


# =========================
# TRAIN
# =========================

def train():

    (X_train, y_train,
     X_val, y_val,
     X_test, y_test,
     scaler, df) = prepare_data()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    model = LSTM(input_size=5).to(device)  # 🔥 IMPORTANTE

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scaler_path = MODEL_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    with mlflow.start_run():

        mlflow.log_params({
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "lr": learning_rate,
            "features": "OHLCV"
        })

        # =========================
        # TRAIN LOOP
        # =========================

        for epoch in range(num_epochs):

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

            avg_loss = total_loss / len(train_loader)

            print(f"Epoch {epoch+1} - Loss: {avg_loss:.6f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

        # =========================
        # VALIDATION
        # =========================

        model.eval()

        with torch.no_grad():
            val_preds = model(X_val.to(device)).cpu().numpy().flatten()
            val_true = y_val.numpy()

        val_mae = mean_absolute_error(val_true, val_preds)
        val_rmse = mean_squared_error(val_true, val_preds) ** 0.5

        print("\nVALIDATION:")
        print(f"MAE: {val_mae:.4f}")
        print(f"RMSE: {val_rmse:.4f}")

        # =========================
        # TEST
        # =========================

        with torch.no_grad():
            test_preds = model(X_test.to(device)).cpu().numpy().flatten()
            test_true = y_test.numpy()

        test_mae = mean_absolute_error(test_true, test_preds)
        test_rmse = mean_squared_error(test_true, test_preds) ** 0.5
        dir_acc = directional_accuracy(test_true, test_preds)

        # =========================
        # BASELINE
        # =========================
        baseline = X_test[:, -1, 3].numpy()  # close anterior

        baseline_mae = mean_absolute_error(test_true, baseline)
        baseline_rmse = mean_squared_error(test_true, baseline) ** 0.5

        print("\nTEST:")
        print(f"MAE: {test_mae:.4f}")
        print(f"RMSE: {test_rmse:.4f}")
        print(f"Directional Accuracy: {dir_acc:.4f}")

        print("\nBASELINE:")
        print(f"MAE: {baseline_mae:.4f}")
        print(f"RMSE: {baseline_rmse:.4f}")

        mlflow.log_metrics({
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "directional_accuracy": dir_acc,
            "baseline_mae": baseline_mae,
            "baseline_rmse": baseline_rmse
        })

        # =========================
        # SAVE MODEL
        # =========================

        model_path = MODEL_DIR / "lstm_ohlcv.pt"
        torch.save(model.state_dict(), model_path)

        sample = X_train[:10].to(device)
        pred_sample = model(sample).detach().cpu().numpy()

        signature = infer_signature(
            X_train[:10].numpy(),
            pred_sample
        )

        mlflow.pytorch.log_model(
            model.cpu(),
            "model",
            signature=signature,
            input_example=X_train[:1].numpy()
        )

        mlflow.log_artifact(str(scaler_path))

        print("\nTreinamento finalizado.")


if __name__ == "__main__":
    train()