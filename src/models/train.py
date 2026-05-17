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

# Seleciona automaticamente GPU se disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hiperparâmetros principais
sequence_length = 60
batch_size = 64
num_epochs = 50
learning_rate = 0.001

# Features utilizadas no modelo
FEATURE_COLUMNS = ["open", "high", "low", "close", "volume"]

# Índice da variável alvo (close)
TARGET_INDEX = FEATURE_COLUMNS.index("close")

# Caminhos do projeto
BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "data/refined/refined.db"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MLFLOW_DIR = BASE_DIR / "mlflow"
MLFLOW_DIR.mkdir(exist_ok=True)

# Configuração do MLflow (usando SQLite como backend)
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
    """
    Carrega os dados históricos de preços a partir de um banco SQLite.

    Returns:
        pd.DataFrame: DataFrame contendo os dados ordenados por data.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM prices", conn)
    conn.close()

    # Padroniza nomes de colunas
    df.columns = df.columns.str.lower().str.strip()

    # Ordena cronologicamente
    df = df.sort_values("date")

    return df


# =========================
# SEQUENCES
# =========================

def create_sequences(data, seq_len):
    """
    Converte uma série temporal em sequências supervisionadas.

    Args:
        data (np.ndarray): Dados normalizados.
        seq_len (int): Tamanho da janela temporal.

    Returns:
        tuple: (X, y)
            X -> (num_samples, seq_len, num_features)
            y -> (num_samples,)
    """
    X, y = [], []

    for i in range(len(data) - seq_len):
        # Sequência de entrada
        X.append(data[i:i+seq_len])

        # Target: próximo valor de "close"
        y.append(data[i+seq_len, TARGET_INDEX])

    return np.array(X), np.array(y)


# =========================
# PREP DATA
# =========================

def prepare_data():
    """
    Realiza todo o pipeline de preparação dos dados:
    - Carregamento
    - Split temporal (train/val/test)
    - Normalização (MinMaxScaler)
    - Criação de sequências
    - Conversão para tensores

    Returns:
        tuple: Tensores de treino, validação, teste e scaler treinado
    """

    df = load_data()

    # Seleciona apenas features relevantes
    features = df[FEATURE_COLUMNS].values

    # Split temporal (sem embaralhamento)
    train_size = int(len(features) * 0.7)
    val_size = int(len(features) * 0.15)

    train = features[:train_size]
    val = features[train_size:train_size+val_size]
    test = features[train_size+val_size:]

    # Normalização (fit apenas no treino -> evita data leakage)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)

    # Criação de sequências temporais
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
    """
    Executa o pipeline completo de treinamento do modelo LSTM.

    Etapas:
    - Preparação dos dados
    - Treinamento com DataLoader
    - Validação por época
    - Early stopping
    - Avaliação em teste (RMSE)
    - Logging completo no MLflow
    - Persistência do modelo e scaler
    """

    global best_val_loss, early_stop_counter

    # =========================
    # DATA
    # =========================
    (X_train, y_train,
     X_val, y_val,
     X_test, y_test,
     scaler) = prepare_data()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    # =========================
    # MODEL
    # =========================
    model = LSTM(input_size=5).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Salva scaler para uso em inferência
    scaler_path = MODEL_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    best_model_path = MODEL_DIR / "lstm_ohlcv.pt"

    # =========================
    # MLFLOW RUN
    # =========================
    with mlflow.start_run():

        # Log de hiperparâmetros
        mlflow.log_params({
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "features": "OHLCV",
            "early_stopping_patience": PATIENCE
        })

        for epoch in range(num_epochs):

            # =========================
            # TRAIN LOOP
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

            # Log métricas no MLflow
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            # =========================
            # EARLY STOPPING
            # =========================
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0

                # Salva melhor modelo
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
        # TEST EVALUATION
        # =========================
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test.to(device)).cpu().numpy().flatten()
            test_true = y_test.numpy()

        # RMSE
        test_rmse = mean_squared_error(test_true, test_preds) ** 0.5

        print(f"\nTest RMSE: {test_rmse:.4f}")

        # Log final no MLflow
        mlflow.log_metric("test_rmse", test_rmse)

        # Salva artefatos
        mlflow.log_artifact(str(best_model_path))
        mlflow.log_artifact(str(scaler_path))

        print("\nTreinamento finalizado.")


if __name__ == "__main__":
    train()