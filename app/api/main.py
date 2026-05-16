from fastapi import FastAPI
from app.api.routes import health, predict, retrain

app = FastAPI()

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(retrain.router)