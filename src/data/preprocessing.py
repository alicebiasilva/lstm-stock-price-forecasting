from pathlib import Path
import pandas as pd
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TABLE_NAME = "prices"


# ==============================
# LOAD
# ==============================

def load_raw_data(raw_path: Path) -> pd.DataFrame:
    logger.info(f"Lendo dados de {raw_path}")
    
    if not raw_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {raw_path}")
    
    df = pd.read_csv(raw_path)
    return df


# ==============================
# TRANSFORM (ESSENCIAL por causa do .stack())
# ==============================

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Transformando dados...")

    # seu stack gera colunas diferentes → vamos normalizar

    # tenta detectar formato automaticamente
    if "level_1" in df.columns:
        # caso vindo do stack
        df = df.rename(columns={
            "level_0": "Date",
            "level_1": "Variable",
            0: "Value"
        })

        # pivotar de volta
        df = df.pivot(index="Date", columns="Variable", values="Value").reset_index()

    # garantir colunas esperadas
    expected = ["Date", "Open", "High", "Low", "Close", "Volume"]

    for col in expected:
        if col not in df.columns:
            raise ValueError(f"Coluna esperada não encontrada: {col}")

    # adicionar ticker (porque seu stack remove)
    df["Ticker"] = "ITUB4.SA"

    # reorder
    df = df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]

    # tipos
    df["Date"] = pd.to_datetime(df["Date"])
    
    return df


# ==============================
# VALIDATION (básico)
# ==============================

def validate_data(df: pd.DataFrame):
    logger.info("Validando dados...")

    if df.isnull().sum().sum() > 0:
        raise ValueError("Dados possuem valores nulos")

    if not (df["Close"] > 0).all():
        raise ValueError("Valores inválidos em Close")


# ==============================
# SAVE SQLITE (REPLACE)
# ==============================

def save_to_sqlite(df: pd.DataFrame, db_path: Path):
    logger.info(f"Salvando no SQLite em {db_path}")

    conn = sqlite3.connect(db_path)

    try:
        conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")

        conn.execute(f"""
            CREATE TABLE {TABLE_NAME} (
                Date TEXT,
                Ticker TEXT,
                Open REAL,
                High REAL,
                Low REAL,
                Close REAL,
                Volume INTEGER
            )
        """)

        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

        logger.info("Dados salvos com sucesso")

    finally:
        conn.close()


# ==============================
# PIPELINE
# ==============================

def run_pipeline(raw_path: Path, db_path: Path):
    df = load_raw_data(raw_path)

    df = transform_data(df)

    validate_data(df)

    save_to_sqlite(df, db_path)

    logger.info("Preprocessing finalizado.")