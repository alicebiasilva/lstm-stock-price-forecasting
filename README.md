# 📈 LSTM Stock Price Forecasting

Projeto de previsão de preços de ações utilizando redes neurais do tipo LSTM, desenvolvido durante a especialização em Engenharia de Machine Learning da FIAP.

*  **Por:** Alice Beatriz da Silva  
*  **Última atualização:** 17/05/2026  

<br>

---

<br>

## Índice
1. Funcionalidades  
2. Estrutura do projeto  
3. Tecnologias  
4. Modelagem e fonte de dados  
5. Instalação e configuração  

<br>

---

<br>

## ⚙️ 1. Funcionalidades

O objetivo deste projeto é construir um **pipeline completo de Machine Learning** para previsão de preços de ativos financeiros, contemplando:

- Treinamento de modelo de séries temporais (LSTM)  
- Pipeline de inferência  
- Exposição via API  
- Observabilidade com MLflow  
- Possibilidade de retreinamento do modelo  

 O projeto simula um cenário real de produção com práticas de **MLOps**.

<br>

---

<br>

## 📂 2. Estrutura do projeto

<br>

O projeto está organizado da seguinte forma:

| Pasta/Arquivo | Descrição |
|--------------|----------|
| `app/` | Camada da API (FastAPI), responsável por expor os endpoints |
| `app/api/routes/` | Definição das rotas da API |
| `data/raw` | Dados brutos provenientes do Yahoo Finance |
| `data/refined` | Banco de dados com dados tratados |
| `data_exploration/` | Notebook de análise exploratória (EDA) |
| `mlflow/` | Configuração do MLflow |
| `mlruns/` | Artefatos e experimentos registrados |
| `models/` | Modelos treinados (.pt e .pkl) |
| `scripts/` | Scripts executáveis do pipeline |
| `src/` | Código fonte principal (data, model, inference) |
| `requirements.txt` | Dependências do projeto |
| `README.md` | Documentação |

<br>

### Benefícios da Arquitetura

- Separação de responsabilidades (treino, inferência, API)  
- Reprodutibilidade com MLflow  
- Escalabilidade para produção  
- Flexibilidade para evolução do modelo  
- Organização modular  


<br>

---

<br>

## 🛠️ 3. Tecnologias Utilizadas

Este projeto foi desenvolvido com foco em boas práticas de Engenharia de Machine Learning e MLOps.

### Stack principal

- **Python** — linguagem principal  
- **PyTorch** — construção e treinamento da LSTM  
- **Scikit-learn** — pré-processamento (MinMaxScaler)  
- **Pandas & NumPy** — manipulação de dados  
- **SQLite** — armazenamento local  
- **FastAPI** — criação da API  
- **Uvicorn** — servidor ASGI  
- **MLflow** — rastreamento de experimentos  

Com o MLflow, é possível:

- Registrar parâmetros do modelo
- Monitorar métricas (loss, RMSE) 
- Comparar versões de modelos 
- Armazenar artefatos 
- Auditar experimentos 

Isso aproxima o projeto de um ambiente real de produção com **governança de modelos**.

### Links úteis

- PyTorch: https://pytorch.org/  
- FastAPI: https://fastapi.tiangolo.com/  
- MLflow: https://mlflow.org/  
- Scikit-learn: https://scikit-learn.org/  

<br>

---

<br>

## 4. Modelagem e fonte de dados

O modelo utiliza dados históricos de ações com as features: Open, High, Low, Close e Volume (OHLCV) via Yahoo Finance (`yfinance`).

<br>

### O que é uma LSTM?

A **LSTM (Long Short-Term Memory)** é um tipo de rede neural recorrente (RNN) capaz de aprender dependências de longo prazo em séries temporais.

Ela utiliza mecanismos de *gates*:

- Gate de entrada  
- Gate de memória (esquecimento)  
- Gate de saída  

Isso permite capturar padrões complexos e evitar problemas como *vanishing gradient*.

<br>

### Estratégia de Modelagem

- Uso de **janelas temporais** (ex: 60 períodos)  
- Entrada: sequências com múltiplas features (OHLCV)  
- Target: close no próximo timestep 
- Formato do input: `(batch_size, sequence_length, num_features)`

<br>

### Pré-processamento

- Normalização com **MinMaxScaler**  
- Ajuste apenas no treino (evita *data leakage*)  
- Aplicação em validação e teste  

<br>

### Hiperparâmetros

- `sequence_length`: 60  
- `input_size`: 5  
- `batch_size`: 64  
- `learning_rate`: 0.001  
- `num_epochs`: 50  
- `early_stopping_patience`: definido no treino  

<br>

### Treinamento

- Divisão temporal (treino, validação, teste)  
- Loss: **MSE**  
- Otimizador: **Adam**  
- Monitoramento: **val_loss**  
- Early stopping para evitar overfitting  

<br>

### Avaliação

- 📉 RMSE (Root Mean Squared Error)  
- 🔁 Comparação com baseline (último valor observado)  

<br>

---

<br>

## ⚙️ 5. Como Executar

### 1. Criar ambiente virtual

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Treinar o modelo

```bash
python -m scripts.run_training
```

### 5. Rodar inferência

```bash
python -m scripts.run_inference
```

### 6. Subir a API

```bash
uvicorn app.api.main:app --reload
```

### Para visualizar experimentos no MLflow
```bash
mlflow ui
```