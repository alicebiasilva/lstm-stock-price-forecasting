from pathlib import Path
from src.data.preprocessing import run_pipeline

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = BASE_DIR / "data/raw/raw.csv"
DB_PATH = BASE_DIR / "data/refined/refined.db"

def main():
    print("Iniciando gravação dos dados no banco de dados...")
    run_pipeline(RAW_PATH, DB_PATH)
    print("Gravação dos dados no banco de dados finalizada!")

if __name__ == "__main__":
    main()