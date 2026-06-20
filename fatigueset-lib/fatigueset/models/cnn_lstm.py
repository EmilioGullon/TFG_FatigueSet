# -*- coding: utf-8 -*-
"""
Híbrido CNN-LSTM en PyTorch implementado manualmente para FatigueSet.
"""

from __future__ import annotations
from typing import Tuple, Optional

try:
    import torch
    import torch.nn as nn
    from .lstm import CustomLSTM
except ImportError:
    torch = None
    nn = None
    CustomLSTM = None


if nn is not None:
    class CustomCNNLSTMRegressor(nn.Module):
        """
        Regresor híbrido CNN-LSTM para regresión fisiológica de fatiga.
        Extrae patrones locales temporales con una Conv1D y MaxPool1d, 
        y posteriormente captura dependencias a largo plazo usando nuestra LSTM manual.
        """
        def __init__(
            self,
            input_size: int,
            conv_channels: int = 64,
            kernel_size: int = 3,
            pool_size: int = 2,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2,
            output_size: int = 2
        ):
            super().__init__()
            self.input_size = input_size
            self.conv_channels = conv_channels
            self.kernel_size = kernel_size
            self.pool_size = pool_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.dropout = dropout
            self.output_size = output_size
            
            # Capa Convolucional 1D: Espera (B, input_size, seq_len)
            self.conv = nn.Conv1d(
                in_channels=input_size,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            )
            self.relu = nn.ReLU()
            
            # Capa de Pooling 1D para reducción temporal (Downsampling)
            self.pool = nn.MaxPool1d(kernel_size=pool_size)
            
            # Capa LSTM manual: Espera (B, new_seq_len, conv_channels)
            self.lstm = CustomLSTM(
                input_size=conv_channels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )
            
            # Capa de salida lineal para regresión bidimensional (física + mental)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            x: Tensor de entrada con dimensiones (batch_size, seq_len, input_size)
            Retorna: Tensor de salida de dimensiones (batch_size, output_size)
            """
            # Transponer x para que tenga el formato de canales esperado por nn.Conv1d: (B, input_size, seq_len)
            x_conv_in = x.transpose(1, 2)
            
            # Aplicar convolución, activación y pooling
            x_conv_out = self.conv(x_conv_in)
            x_act = self.relu(x_conv_out)
            x_pool = self.pool(x_act)  # Dimensión: (B, conv_channels, new_seq_len)
            
            # Transponer de vuelta para que tenga el formato de secuencia esperado por la LSTM: (B, new_seq_len, conv_channels)
            x_lstm_in = x_pool.transpose(1, 2)
            
            # Procesar secuencia temporal comprimida mediante LSTM
            out, _ = self.lstm(x_lstm_in)
            
            # Tomar el último paso temporal de la secuencia reducida
            last_step_out = out[:, -1, :]
            
            # Regresión lineal final
            return self.fc(last_step_out)

else:
    class CustomCNNLSTMRegressor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch no está disponible. Instale torch para usar CustomCNNLSTMRegressor.")
