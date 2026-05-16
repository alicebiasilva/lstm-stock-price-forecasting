from flask import Flask
from flasgger import Swagger
from app.api.routes import api_bp
from app.api.config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # 🔹 Inicializa Swagger
    Swagger(app)

    # 🔹 Registra rotas
    app.register_blueprint(api_bp, url_prefix="/api")

    return app