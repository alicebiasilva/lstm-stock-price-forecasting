from src.inference.predict import run_inference


def main():
    """
    Função principal para execução do pipeline de inferência.

    Etapas:
    - Executa o processo de previsão utilizando o modelo treinado
    - Exibe o resultado final formatado
    - Trata possíveis erros durante a execução

    Essa função serve como ponto de entrada para rodar inferência
    via linha de comando de forma simples e controlada.
    """

    try:
        # Executa pipeline de inferência
        result = run_inference()

        # Exibe resultado formatado
        print(f"\nResultado final: {result:.4f}")

    except Exception as e:
        # Tratamento genérico de erro para evitar quebra do script
        print(f"\nErro durante inferência: {str(e)}")


if __name__ == "__main__":
    # Garante execução apenas quando chamado diretamente
    main()