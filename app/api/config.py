class Config:
    # Chave secreta usada para segurança da aplicação (ex: sessões, JWT, CSRF, etc.)
    SECRET_KEY = "sua_chave_secreta"

    # Tipo de cache utilizado pela aplicação. "simple" é o cache local em memória (útil para dev)
    CACHE_TYPE = "simple"

    # Título exibido na documentação Swagger (OpenAPI)
    SWAGGER_TITLE = "Book API - Tech Challenge"

    # Versão da API mostrada na documentação Swagger
    SWAGGER_VERSION = "1.0.0"

    # Desativa o rastreamento de modificações de objetos no SQLAlchemy
    # Isso economiza recursos, pois esse recurso raramente é necessário
    SQLALCHEMY_TRACK_MODIFICATIONS = False