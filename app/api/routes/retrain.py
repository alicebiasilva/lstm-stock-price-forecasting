from fastapi import APIRouter
from src.models.train import train

# Instância do roteador da API
router = APIRouter()


@router.post("/retrain")
def retrain():
    """
    Endpoint responsável por disparar o retreinamento do modelo.

    Fluxo:
    - Executa o pipeline completo de treinamento
    - Atualiza o modelo salvo em disco
    - (Opcional) registra nova versão no MLflow

    Returns:
        dict: Status da operação.

    ⚠️ Observação:
        Este endpoint executa uma tarefa pesada (treinamento),
        podendo impactar performance da API. Em produção, o ideal
        seria delegar essa tarefa para um worker assíncrono
        (ex: Celery, fila de jobs, etc).
    """

    # Executa o processo de treinamento do modelo
    train()

    # Retorna confirmação simples
    return {"status": "retrained"}