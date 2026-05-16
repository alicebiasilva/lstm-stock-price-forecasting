from src.inference.predict import run_inference


def main():
    try:
        result = run_inference()
        print(f"\nResultado final: {result:.4f}")

    except Exception as e:
        print(f"\nErro durante inferência: {str(e)}")


if __name__ == "__main__":
    main()