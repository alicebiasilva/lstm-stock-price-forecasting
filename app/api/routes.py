from flask import Blueprint, request, jsonify
import threading
import numpy as np

# CORE do projeto (MLOps style)
from src.pipeline.train_pipeline import run_training_pipeline
from src.inference.predict import predict as model_predict
# from src.models.load_model import load_model  # opcional

api_bp = Blueprint("api", __name__)

# =========================================================
# MODEL (lazy loading)
# =========================================================
model = None

def get_model():
    global model
    if model is None:
        # Exemplo (ajuste conforme seu projeto)
        # model = load_model("models/itau_lstm_v1.h5")
        pass
    return model


# =========================================================
# 1. HEALTH CHECK
# =========================================================
@api_bp.route("/health", methods=["GET"])
def health():
    """
    Health Check da API
    ---
    responses:
      200:
        description: API funcionando corretamente
    """
    return jsonify({
        "status": "ok",
        "message": "API está ativa"
    })


# =========================================================
# 2. PREDICT
# =========================================================
@api_bp.route("/predict", methods=["POST"])
def predict():
    """
    Previsão de preços (LSTM)
    ---
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - input_data
          properties:
            input_data:
              type: array
              items:
                type: number
    responses:
      200:
        description: Previsão realizada com sucesso
      400:
        description: Erro de validação
    """

    try:
        data = request.get_json()

        if not data or "input_data" not in data:
            return jsonify({"error": "Campo 'input_data' é obrigatório"}), 400

        input_data = np.array(data["input_data"])

        # Ajuste para LSTM
        input_data = input_data.reshape(1, input_data.shape[0], 1)

        model_instance = get_model()

        forecast = model_predict(model_instance, input_data)

        return jsonify({
            "forecast_10_days": forecast
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================================================
# 3. RETRAIN (background)
# =========================================================
def retrain_pipeline():
    global model

    try:
        print("🚀 Iniciando retreino...")

        run_training_pipeline()

        # força reload do modelo após treino
        model = None

        print("✅ Retreino finalizado!")

    except Exception as e:
        print(f"❌ Erro no retreino: {str(e)}")


@api_bp.route("/retrain", methods=["POST"])
def retrain():
    """
    Dispara retreino do modelo
    ---
    responses:
      202:
        description: Retreino iniciado
    """

    thread = threading.Thread(target=retrain_pipeline)
    thread.start()

    return jsonify({
        "status": "started",
        "message": "Retreino iniciado em background"
    }), 202