# -*- coding: utf-8 -*-
"""
Temporal Convolutional Network (TCN) en PyTorch implementado de forma manual para FatigueSet.
"""

from __future__ import annotations
from typing import List, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None


if nn is not None:
    class TemporalBlock(nn.Module):
        """
        Bloque residual para TCN consistente en dos capas de convolución causal dilatada
        con activación ReLU, Weight Normalization y Dropout.
        """
        def __init__(
            self,
            n_inputs: int,
            n_outputs: int,
            kernel_size: int,
            stride: int,
            dilation: int,
            dropout: float = 0.2
        ):
            super().__init__()
            self.kernel_size = kernel_size
            self.dilation = dilation
            self.padding = (kernel_size - 1) * dilation
            
            # Convolución 1 + Weight Norm
            # Usamos padding=0 en nn.Conv1d para aplicar el padding causal manualmente en forward
            self.conv1 = nn.utils.weight_norm(nn.Conv1d(
                in_channels=n_inputs,
                out_channels=n_outputs,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                dilation=dilation
            ))
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)
            
            # Convolución 2 + Weight Norm
            self.conv2 = nn.utils.weight_norm(nn.Conv1d(
                in_channels=n_outputs,
                out_channels=n_outputs,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                dilation=dilation
            ))
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)
            
            # Capa de proyección 1x1 si las dimensiones de canales de entrada y salida difieren
            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
            self.relu_out = nn.ReLU()
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (B, n_inputs, L)
            
            # Primera capa: Padding causal + Conv1d + ReLU + Dropout
            x1_padded = F.pad(x, (self.padding, 0), mode='constant', value=0.0)
            out = self.conv1(x1_padded)
            out = self.relu1(out)
            out = self.dropout1(out)
            
            # Segunda capa: Padding causal + Conv1d + ReLU + Dropout
            out_padded = F.pad(out, (self.padding, 0), mode='constant', value=0.0)
            out = self.conv2(out_padded)
            out = self.relu2(out)
            out = self.dropout2(out)
            
            # Conexión residual
            res = x if self.downsample is None else self.downsample(x)
            
            return self.relu_out(out + res)


    class TemporalConvNet(nn.Module):
        """
        Cadena de bloques residuales temporales (TCN).
        Multiplica la dilación por 2 en cada nivel consecutivo para expandir
        exponencialmente el campo receptivo.
        """
        def __init__(self, num_inputs: int, num_channels: list[int], kernel_size: int = 3, dropout: float = 0.2):
            super().__init__()
            layers = []
            num_levels = len(num_channels)
            for i in range(num_levels):
                dilation_size = 2 ** i
                in_channels = num_inputs if i == 0 else num_channels[i - 1]
                out_channels = num_channels[i]
                layers.append(TemporalBlock(
                    n_inputs=in_channels,
                    n_outputs=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    dropout=dropout
                ))
            self.network = nn.Sequential(*layers)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)


    class CustomTCNRegressor(nn.Module):
        """
        Regresor basado en Temporal Convolutional Network (TCN).
        Recibe un tensor con forma (batch_size, seq_len, input_size),
        lo permuta para el procesamiento convolucional a (batch_size, input_size, seq_len),
        aplica la red TCN, extrae el estado causal final en t = -1 (el último timestep),
        y genera la predicción mediante una capa lineal de salida.
        """
        def __init__(
            self,
            input_size: int,
            num_channels: list[int] = [64, 64, 64, 64, 64],
            kernel_size: int = 3,
            dropout: float = 0.2,
            output_size: int = 2
        ):
            super().__init__()
            self.input_size = input_size
            self.num_channels = num_channels
            self.kernel_size = kernel_size
            self.dropout = dropout
            self.output_size = output_size
            
            self.tcn = TemporalConvNet(
                num_inputs=input_size,
                num_channels=num_channels,
                kernel_size=kernel_size,
                dropout=dropout
            )
            
            # Capa lineal final para regresión
            self.fc = nn.Linear(num_channels[-1], output_size)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (B, seq_len, input_size)
            # Permutar a: (B, input_size, seq_len)
            x_in = x.transpose(1, 2)
            
            # Procesar con TCN
            tcn_out = self.tcn(x_in) # shape: (B, num_channels[-1], seq_len)
            
            # Dado que las convoluciones son causales, el último paso temporal (t = -1)
            # contiene la información resumida de toda la secuencia sin fugas.
            last_step = tcn_out[:, :, -1] # shape: (B, num_channels[-1])
            
            # Mapear a la salida bidimensional (fatiga física, fatiga mental)
            return self.fc(last_step)

else:
    class CustomTCNRegressor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch no está disponible. Instale torch para usar CustomTCNRegressor.")
