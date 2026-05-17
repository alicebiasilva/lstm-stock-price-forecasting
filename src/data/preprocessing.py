from pathlib import Path
import pandas as pd
import sqlite3
import logging

# Configuração básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Nome da tabela no SQLite
TABLE_NAME = "prices"


# ==============================
# LOAD
# ==============================

def load_raw_data(raw_path: Path) -> pd.DataFrame:
    """
    Carrega os dados brutos a partir de um arquivo CSV.

    Args:
        raw_path (Path): Caminho para o arquivo CSV contendo dados raw.

    Returns:
        pd.DataFrame: DataFrame com os dados carregados.

    Raises:
        FileNotFoundError: Caso o arquivo não exista.
    """
    logger.info(f"Lendo dados de {raw_path}")
    
    # Validação de existência do arquivo
    if not raw_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {raw_path}")
    
    df = pd.read_csv(raw_path)
    return df


# ==============================
# TRANSFORM
# ==============================

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza transformações necessárias nos dados brutos para padronização.

    Etapas:
    - Ajusta formato vindo de operações como .stack()
    - Normaliza nomes de colunas
    - Garante presença de colunas esperadas
    - Adiciona coluna de ticker
    - Converte tipos

    Args:
        df (pd.DataFrame): DataFrame bruto.

    Returns:
        pd.DataFrame: DataFrame transformado e estruturado.
    """
    logger.info("Transformando dados...")

    # =========================
    # HANDLE STACK FORMAT
    # =========================
    # Caso o DataFrame venha do .stack(), ele terá colunas como:
    # level_0, level_1 e valores em uma única coluna
    if "level_1" in df.columns:
        df = df.rename(columns={
            "level_0": "Date",
            "level_1": "Variable",
            0: "Value"
        })

        # Converte de formato longo para largo (pivot)
        df = df.pivot(index="Date", columns="Variable", values="Value").reset_index()

    # =========================
    # VALIDATE EXPECTED COLUMNS
    # =========================
    expected = ["Date", "Open", "High", "Low", "Close", "Volume"]

    for col in expected:
        if col not in df.columns:
            raise ValueError(f"Coluna esperada não encontrada: {col}")

    # =========================
    # ADD METADATA
    # =========================
    # Adiciona ticker (perdido durante transformações)
    df["Ticker"] = "ITUB4.SA"

    # Reorganiza colunas para padrão consistente
    df = df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]

    # =========================
    # TYPE CASTING
    # =========================
    # Converte coluna de data para datetime
    df["Date"] = pd.to_datetime(df["Date"])
    
    return df


# ==============================
# VALIDATION
# ==============================

def validate_data(df: pd.DataFrame):
    """
    Executa validações básicas de qualidade dos dados.

    Validações:
    - Ausência de valores nulos
    - Valores positivos na coluna Close

    Args:
        df (pd.DataFrame): DataFrame a ser validado.

    Raises:
        ValueError: Caso alguma validação falhe.
    """
    logger.info("Validando dados...")

    # Verifica valores nulos
    if df.isnull().sum().sum() > 0:
        raise ValueError("Dados possuem valores nulos")

    # Verifica valores inválidos de preço
    if not (df["Close"] > 0).all():
        raise ValueError("Valores inválidos em Close")


# ==============================
# SAVE SQLITE
# ==============================

def save_to_sqlite(df: pd.DataFrame, db_path: Path):
    """
    Persiste os dados processados em um banco SQLite.

    Estratégia:
    - Remove tabela existente (DROP)
    - Cria nova tabela com schema definido
    - Insere dados com replace

    Args:
        df (pd.DataFrame): Dados processados.
        db_path (Path): Caminho do banco SQLite.
    """
    logger.info(f"Salvando no SQLite em {db_path}")

    conn = sqlite3.connect(db_path)

    try:
        # Remove tabela anterior (evita duplicidade/inconsistência)
        conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")

        # Cria tabela com schema explícito
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

        # Insere dados
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

        logger.info("Dados salvos com sucesso")

    finally:
        conn.close()


# ==============================
# PIPELINE
# ==============================

def run_pipeline(raw_path: Path, db_path: Path):
    """
    Executa o pipeline completo de preprocessing de dados.

    Etapas:
    1. Carregamento dos dados brutos
    2. Transformação e padronização
    3. Validação de qualidade
    4. Persistência no banco SQLite

    Args:
        raw_path (Path): Caminho do CSV bruto.
        db_path (Path): Caminho do banco SQLite.
    """

    # Load
    df = load_raw_data(raw_path)

    # Transform
    df = transform_data(df)

    # Validate
    validate_data(df)

    # Save
    save_to_sqlite(df, db_path)

    logger.info("Preprocessing finalizado.")