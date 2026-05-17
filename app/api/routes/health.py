from fastapi import APIRouter

# Instância do roteador da API
router = APIRouter()


@router.get("/health")
def health():
    """
    Endpoint de health check da API.

    Utilizado para verificar se a aplicação está ativa e respondendo corretamente.
    Muito usado em:
    - Monitoramento (ex: Kubernetes, load balancers)
    - Testes automatizados
    - Observabilidade

    Returns:
        dict: Status simples indicando que a API está saudável.
    """

    # Retorna resposta padrão de sucesso
    return {"status": "ok"}