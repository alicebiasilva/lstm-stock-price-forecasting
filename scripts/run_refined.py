from pathlib import Path
from src.data.preprocessing import run_pipeline

# Diretório base do projeto (um nível acima do script atual)
BASE_DIR = Path(__file__).resolve().parents[1]

# Caminho do arquivo CSV bruto
RAW_PATH = BASE_DIR / "data/raw/raw.csv"

# Caminho do banco de dados refinado
DB_PATH = BASE_DIR / "data/refined/refined.db"


def main():
    """
    Função principal responsável por executar o pipeline de preprocessing.

    Etapas:
    - Lê os dados brutos (CSV)
    - Executa transformação e validação
    - Persiste os dados no banco SQLite

    Esse script atua como ponto de entrada para a etapa de
    preparação de dados dentro do pipeline de Machine Learning.
    """

    # Log de início do processo
    print("Iniciando gravação dos dados no banco de dados...")

    # Executa pipeline completo de preprocessing
    run_pipeline(RAW_PATH, DB_PATH)

    # Log de finalização
    print("Gravação dos dados no banco de dados finalizada!")


if __name__ == "__main__":
    # Garante execução apenas quando o script for chamado diretamente
    main()