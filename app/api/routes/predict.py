from fastapi import APIRouter
from src.inference.predict import run_inference

router = APIRouter()

@router.get("/predict")
def predict():
    return {"prediction": run_inference()}