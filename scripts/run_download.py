from src.data.download import download_data


def main():
    """
    Função principal responsável por executar o pipeline de download de dados.

    Etapas:
    - Dispara o processo de extração de dados do Yahoo Finance
    - Exibe logs simples de início e fim da execução

    Essa função atua como ponto de entrada do script, facilitando
    execução via linha de comando.
    """

    # Log de início do processo
    print("Iniciando download...")

    # Executa função de download definida no módulo de dados
    download_data()

    # Log de conclusão
    print("Download finalizado!")


if __name__ == "__main__":
    # Garante que o script só será executado diretamente,
    # evitando execução automática quando importado como módulo
    main()