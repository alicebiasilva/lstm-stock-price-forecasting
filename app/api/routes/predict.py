from fastapi import APIRouter
from src.inference.predict import run_inference

# Instância do roteador da API
router = APIRouter()


@router.get("/predict")
def predict():
    """
    Endpoint responsável por realizar a inferência do modelo.

    Fluxo:
    - Executa o pipeline de inferência
    - Retorna a previsão do próximo valor da série temporal

    Returns:
        dict: Dicionário contendo o valor previsto.
    
    Exemplo de resposta:
        {
            "prediction": 32.45
        }
    """

    # Executa função de inferência (carrega modelo, dados e gera previsão)
    prediction = run_inference()

    # Retorna resultado em formato JSON
    return {"prediction": prediction}