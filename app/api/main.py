import time
from fastapi import FastAPI, Request

from app.api.routes import health, predict, retrain

# Instância principal da aplicação FastAPI
app = FastAPI()


@app.middleware("http")
async def log_request_time(request: Request, call_next):
    """
    Middleware para logging do tempo de resposta das requisições.

    Fluxo:
    - Captura o tempo inicial da requisição
    - Executa o processamento da rota
    - Calcula o tempo total de execução
    - Loga método, endpoint e tempo de resposta

    Args:
        request (Request): Objeto da requisição HTTP
        call_next: Função que encaminha a requisição para o próximo handler

    Returns:
        Response: Resposta da API após processamento
    """

    # Marca início da requisição
    start_time = time.time()

    # Processa requisição (chama endpoint correspondente)
    response = await call_next(request)

    # Calcula tempo total de execução
    process_time = time.time() - start_time

    # Log simples (pode ser substituído por logging estruturado)
    print(f"{request.method} {request.url.path} - {process_time:.4f}s")

    return response


# =========================
# ROUTES REGISTRATION
# =========================

# Health check endpoint (/health)
app.include_router(health.router)

# Endpoint de inferência (/predict)
app.include_router(predict.router)

# Endpoint de retreinamento (/retrain)
app.include_router(retrain.router)