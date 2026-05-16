from fastapi import APIRouter
from src.models.train import train

router = APIRouter()

@router.post("/retrain")
def retrain():
    train()
    return {"status": "retrained"}