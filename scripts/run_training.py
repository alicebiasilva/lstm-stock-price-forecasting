from src.models.train import train


if __name__ == "__main__":
    """
    Ponto de entrada para execução do treinamento do modelo.

    Ao rodar este script diretamente:
    - Executa o pipeline de treinamento definido em src.models.train
    - Realiza todas as etapas: preparação de dados, treino, validação e logging

    Uso:
        python -m scripts.run_training
    """

    # Chama função principal de treinamento
    train()