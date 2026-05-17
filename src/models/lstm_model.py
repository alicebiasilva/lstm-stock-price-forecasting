import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    Modelo de rede neural recorrente do tipo LSTM para previsão de séries temporais.

    Esta arquitetura recebe como entrada uma sequência temporal multivariada
    e aprende padrões ao longo do tempo para prever o próximo valor da série.

    Args:
        input_size (int): Número de features de entrada (ex: 5 para OHLCV).
        hidden_size (int): Número de unidades ocultas na LSTM.
        num_layers (int): Número de camadas LSTM empilhadas.
        output_size (int): Dimensão da saída (ex: 1 para previsão de preço).

    Input shape:
        x -> (batch_size, sequence_length, input_size)

    Output shape:
        out -> (batch_size, output_size)
    """

    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super().__init__()

        # Número de unidades ocultas da LSTM
        self.hidden_size = hidden_size

        # Número de camadas empilhadas
        self.num_layers = num_layers

        # Camada LSTM principal
        # batch_first=True => entrada no formato (batch, seq, feature)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Camada totalmente conectada para mapear saída da LSTM -> previsão final
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Executa o forward pass da rede.

        Args:
            x (torch.Tensor): Tensor de entrada com shape
                              (batch_size, sequence_length, input_size)

        Returns:
            torch.Tensor: Previsão do modelo com shape (batch_size, output_size)
        """

        # Inicializa estados ocultos (hidden state) com zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Inicializa estados de célula (cell state) com zeros
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward na LSTM
        # out -> (batch_size, sequence_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # Seleciona apenas o último timestep da sequência
        # out[:, -1, :] -> (batch_size, hidden_size)
        out = out[:, -1, :]

        # Passa pela camada linear para gerar a previsão final
        out = self.fc(out)

        return out