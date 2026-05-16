# Previsão de Preços Futuros de Ações (ITUB4)

--- 

-- mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db

[APAGAR]
Não esquecer de explicar:

Separação de responsabilidades
src → lógica pura - core ✔️
scripts → execução, só chamam o core✔️
app → serving ✔️

project/
│
├── app/
│   └── api/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── pipeline/        
│   │   └── train_pipeline.py
│   ├── inference/       
│   │   └── predict.py
│
├── scripts/
│   ├── run_download.py
│   ├── run_preprocessing.py
│   └── run_training.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/              
│
├── run.py

--- 

Este projeto desenvolve um modelo de Deep Learning baseado em LSTM (Long Short-Term Memory) para prever os 10 próximos valores de fechamento diário da ação ITUB4.SA (Itaú Unibanco), usando dados históricos de mercado.

O sistema inclui todo o fluxo de Machine Learning, desde a coleta de dados até o deploy de uma API REST para inferência.

* Autora: Alice Beatriz da Silva
* Última atualização: 09/05/2026
* Deploy disponível em: ...
* Apresentação em vídeo em: ...

<br> 

### Sumário:

1. Sobre a escolha do modelo
2. Funcionalidades e exemplo de uso
3. Pipeline e estrutura do projeto
4. Tecnologias Utilizadas
5. Como Executar Localmente 

<br>
 
---

<br> 

## 1. Sobre a escolha do modelo 
Redes neurais recorrentes padrão (RNNs) têm dificuldade em memorizar informações por longos períodos.  Elas “esquecem” rapidamente eventos antigos, o que prejudica tarefas como previsão de ações, tradução de texto ou reconhecimento de fala.

As LSTMs resolvem isso usando estruturas de memória internas chamadas gates (portas), que decidem:

* O que lembrar
* O que esquecer
* O que usar para a saída

<br>

--- 

<br>

## 2. Pipeline do Projeto


```text
Usuário envia ticker
        ↓
API consulta Yahoo Finance
        ↓
Obtém últimos 60 fechamentos
        ↓
Normaliza dados
        ↓
Modelo LSTM
        ↓
Predição dos próximos 10 dias
        ↓
API retorna JSON
```

---

## Tecnologias Utilizadas

- **Python 3.11**  
- **TensorFlow / Keras** (LSTM)  
- **FastAPI** (API REST)  
- **yfinance** (coleta de dados)  
- **Pandas / NumPy** (manipulação de dados)  
- **Scikit-learn** (pré-processamento e métricas)  
- **MLflow** (rastreio de experimentos)  
- **Docker** (containerização)  
- **Vercel** (deploy da API)  

---

## Estrutura do Projeto

```text
tech-challenge-lstm/
│
├── app/
│   ├── api/          # endpoints FastAPI
│   ├── services/     # lógica do modelo
│   └── model/        # modelo treinado e scaler
│
├── training/         # scripts de treino e avaliação
├── notebooks/        # notebooks de exploração
├── tests/            # testes automatizados
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## API

### Endpoint principal

**POST /predict**

#### Request

```json
{
  "ticker": "ITUB4.SA"
}
```

#### Response

```json
{
  "ticker": "ITUB4.SA",
  "predicted_close": [36.82, 36.95, 37.10, 36.88, 36.75, 36.90, 37.05, 37.20, 37.30, 37.15],
  "prediction_dates": [
    "2026-05-10",
    "2026-05-11",
    "2026-05-12",
    "2026-05-13",
    "2026-05-14",
    "2026-05-15",
    "2026-05-16",
    "2026-05-17",
    "2026-05-18",
    "2026-05-19"
  ]
}
```

---

## Como Executar Localmente

1. Criar ambiente virtual (opcional):

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows
```

2. Instalar dependências:

```bash
pip install -r requirements.txt
```

3. Executar API:

```bash
uvicorn app.api.main:app --reload
```

---

## Docker

1. Build da imagem:

```bash
docker build -t stock-api .
```

2. Rodar container:

```bash
docker run -p 8000:8000 stock-api
```

A API ficará disponível em `http://localhost:8000`.

---

## Observações

- O modelo prevê apenas o **fechamento diário da ação ITUB4**.  
- Os próximos 10 dias são estimativas baseadas em histórico recente.  
- Melhorias futuras incluem multi-ativo, previsão de múltiplos passos mais avançada, cache de dados e monitoramento em produção.