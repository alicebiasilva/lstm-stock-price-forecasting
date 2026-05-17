from src.data.download import download_data
from src.data.preprocessing import run_pipeline
from src.models.train import train


def run_training_pipeline():
    """
    Executa o pipeline completo de treinamento do modelo.

    Etapas:
    - Download dos dados brutos (data ingestion)
    - Processamento e validação dos dados (data preprocessing)
    - Treinamento do modelo LSTM

    Returns:
        model: Modelo treinado.

    Observação:
        Essa função centraliza o fluxo de treinamento, facilitando integração
        com APIs, agendadores (Airflow/Prefect) ou pipelines automatizados.
    """

    # =========================
    # DATA INGESTION
    # =========================
    # Realiza download dos dados históricos
    data = download_data()

    # =========================
    # DATA PREPROCESSING
    # =========================
    # Executa transformação e validação dos dados
    processed = run_pipeline(data)

    # =========================
    # MODEL TRAINING
    # =========================
    # Treina o modelo com os dados processados
    model = train(processed)

    return model