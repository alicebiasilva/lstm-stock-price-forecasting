import os
import yfinance as yf
from datetime import datetime

def download_data(ticker="ITUB4.SA", start="2018-01-01", end=None):
    # Caminho correto na raiz do projeto
    raw_folder = os.path.join("data", "raw")
    os.makedirs(raw_folder, exist_ok=True)

    # Baixa os dados
    df = yf.download(ticker, start=start, end=end)
    df = df.stack(level=1, future_stack=True).reset_index()

    # Salva o CSV na pasta correta
    file_path = os.path.join(raw_folder, f"raw.csv")
    df.to_csv(file_path)

    print(f"Dados de {ticker} salvos em {file_path}")
    return df

if __name__ == "__main__":
    download_data()