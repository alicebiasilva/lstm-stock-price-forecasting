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

# Diretório base do projeto
BASE_DIR = Path(__file__).resolve().parents[2]

# Caminhos dos artefatos
MODEL_PATH = BASE_DIR / "models/lstm_ohlcv.pt"
SCALER_PATH = BASE_DIR / "models/scaler.pkl"
DB_PATH = BASE_DIR / "data/refined/refined.db"

# Features utilizadas (mesmas do treinamento)
FEATURE_COLUMNS = ["open", "high", "low", "close", "volume"]

# Índice da variável alvo (close)
TARGET_INDEX = FEATURE_COLUMNS.index("close")

# Tamanho da sequência usada como entrada para previsão
SEQUENCE_LENGTH = 20

# Seleção automática de dispositivo (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# MLFLOW CONFIG
# =========================

# Configuração do tracking do MLflow
MLFLOW_DIR = BASE_DIR / "mlflow"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR}/mlflow.db")
mlflow.set_experiment("lstm_stock_forecasting_ohlcv")


# =========================
# LOAD DATA
# =========================

def load_data():
    """
    Carrega os dados históricos do banco SQLite.

    Returns:
        pd.DataFrame: Dados ordenados cronologicamente.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM prices", conn)
    conn.close()

    # Padroniza nomes de colunas
    df.columns = df.columns.str.lower().str.strip()

    # Ordena por data (importante para séries temporais)
    df = df.sort_values("date")

    return df


# =========================
# LOAD MODEL + SCALER
# =========================

def load_model_and_scaler():
    """
    Carrega o modelo treinado e o scaler utilizado no pré-processamento.

    Returns:
        tuple:
            model (torch.nn.Module): Modelo LSTM carregado
            scaler (MinMaxScaler): Scaler ajustado no treino
    """

    # Carrega scaler (necessário para normalização e inversão)
    scaler = joblib.load(SCALER_PATH)

    # Instancia o modelo com mesma arquitetura do treinamento
    model = LSTM(input_size=5).to(device)

    # Carrega pesos treinados
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # Coloca em modo avaliação (desativa dropout, etc.)
    model.eval()

    return model, scaler


# =========================
# INFERENCE
# =========================

def run_inference():
    """
    Executa o pipeline de inferência do modelo.

    Etapas:
    - Carrega modelo e scaler
    - Obtém dados mais recentes
    - Normaliza os dados
    - Cria sequência de entrada
    - Realiza previsão
    - Reverte escala (inverse transform)
    - Registra resultado no MLflow

    Returns:
        float: Valor previsto para o próximo timestep (preço de fechamento)
    """

    print("\nIniciando inferência...")

    # Carrega modelo e scaler
    model, scaler = load_model_and_scaler()

    # Carrega dados mais recentes
    df = load_data()
    values = df[FEATURE_COLUMNS].values

    # Normaliza usando scaler do treino
    scaled = scaler.transform(values)

    # Seleciona última janela temporal
    last_seq = scaled[-SEQUENCE_LENGTH:]

    # Ajusta formato para o modelo (batch_size=1)
    X = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)

    # Inicia run no MLflow para rastreamento da inferência
    with mlflow.start_run(run_name="inference_run"):

        print("Rodando previsão...")

        # Desativa cálculo de gradiente (inferência)
        with torch.no_grad():
            pred = model(X).cpu().numpy()

        # =========================
        # INVERSE SCALING
        # =========================
        # Como o scaler foi treinado com múltiplas features,
        # criamos um vetor "dummy" para aplicar inverse_transform corretamente
        dummy = np.zeros((pred.shape[0], 5))

        # Insere previsão na posição correta (close)
        dummy[:, TARGET_INDEX] = pred[:, 0]

        # Reverte escala para valor real
        pred_real = scaler.inverse_transform(dummy)[:, TARGET_INDEX]
        predicted_value = float(pred_real[0])

        print(f"\nPrevisão próxima: {predicted_value:.4f}")

        # Log da previsão no MLflow
        mlflow.log_metric("predicted_price", predicted_value)

    return predicted_value


# =========================
# RUN
# =========================

if __name__ == "__main__":
    run_inference()