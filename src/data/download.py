import os
import yfinance as yf
from datetime import datetime


def download_data(ticker="ITUB4.SA", start="2018-01-01", end=None):
    """
    Realiza o download de dados históricos de ações via Yahoo Finance
    e salva os dados brutos em formato CSV.

    Args:
        ticker (str): Código do ativo (ex: "ITUB4.SA").
        start (str): Data inicial no formato YYYY-MM-DD.
        end (str, optional): Data final. Se None, utiliza a data atual.

    Returns:
        pd.DataFrame: DataFrame contendo os dados históricos baixados.

    Etapas:
        - Criação do diretório de dados brutos
        - Download dos dados via yfinance
        - Transformação do formato do DataFrame
        - Persistência em CSV
    """

    # =========================
    # PATH CONFIG
    # =========================

    # Define pasta de saída para dados brutos
    raw_folder = os.path.join("data", "raw")

    # Cria diretório caso não exista
    os.makedirs(raw_folder, exist_ok=True)

    # =========================
    # DOWNLOAD DATA
    # =========================

    # Se end não for informado, usa data atual
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    # Baixa dados do Yahoo Finance
    df = yf.download(ticker, start=start, end=end)

    # =========================
    # DATA TRANSFORMATION
    # =========================

    # Ajusta formato do DataFrame (caso venha multi-index)
    # stack transforma colunas multi-index em linhas
    df = df.stack(level=1, future_stack=True).reset_index()

    # =========================
    # SAVE DATA
    # =========================

    # Define caminho do arquivo
    file_path = os.path.join(raw_folder, "raw.csv")

    # Salva como CSV
    df.to_csv(file_path, index=False)

    print(f"Dados de {ticker} salvos em {file_path}")

    return df


if __name__ == "__main__":
    download_data()