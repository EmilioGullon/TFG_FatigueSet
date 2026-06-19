# -*- coding: utf-8 -*-
"""
Gated Recurrent Unit (GRU) en PyTorch implementada manualmente para FatigueSet.
"""

from __future__ import annotations
import math
from typing import Tuple, Optional

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


if nn is not None:
    class CustomGRUCell(nn.Module):
        """
        Celda GRU manual que computa las ecuaciones clásicas de Cho et al. (2014).
        """
        def __init__(self, input_size: int, hidden_size: int):
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size

            # Parámetros para la puerta de actualización (Update Gate)
            self.W_z = nn.Parameter(torch.empty(hidden_size, input_size))
            self.U_z = nn.Parameter(torch.empty(hidden_size, hidden_size))
            self.b_z = nn.Parameter(torch.empty(hidden_size))

            # Parámetros para la puerta de reinicio (Reset Gate)
            self.W_r = nn.Parameter(torch.empty(hidden_size, input_size))
            self.U_r = nn.Parameter(torch.empty(hidden_size, hidden_size))
            self.b_r = nn.Parameter(torch.empty(hidden_size))

            # Parámetros para el estado oculto candidato (Candidate Hidden State)
            self.W_h = nn.Parameter(torch.empty(hidden_size, input_size))
            self.U_h = nn.Parameter(torch.empty(hidden_size, hidden_size))
            self.b_h = nn.Parameter(torch.empty(hidden_size))

            self.reset_parameters()

        def reset_parameters(self):
            """Inicialización uniforme estándar (Kaiming/Uniform)."""
            stdv = 1.0 / math.sqrt(self.hidden_size)
            for p in self.parameters():
                p.data.uniform_(-stdv, stdv)

        def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
            """
            Paso de tiempo individual de la celda GRU.
            x: (batch_size, input_size)
            h_prev: (batch_size, hidden_size)
            """
            z_t = torch.sigmoid(x @ self.W_z.t() + h_prev @ self.U_z.t() + self.b_z)
            r_t = torch.sigmoid(x @ self.W_r.t() + h_prev @ self.U_r.t() + self.b_r)
            
            # La puerta de reinicio controla el impacto del estado oculto anterior en el candidato
            h_candidate = torch.tanh(x @ self.W_h.t() + (r_t * h_prev) @ self.U_h.t() + self.b_h)
            
            # Mezcla lineal del estado anterior y el candidato
            h_t = (1.0 - z_t) * h_prev + z_t * h_candidate
            
            return h_t


    class CustomGRU(nn.Module):
        """
        Capa GRU multicapa que procesa secuencias completas usando CustomGRUCell.
        """
        def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0):
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.dropout = dropout

            self.layers = nn.ModuleList()
            for i in range(num_layers):
                layer_input_size = input_size if i == 0 else hidden_size
                self.layers.append(CustomGRUCell(layer_input_size, hidden_size))

            self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        def forward(self, x: torch.Tensor, hx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Procesamiento de toda la secuencia temporal.
            x: (batch_size, seq_len, input_size)
            hx: tensor de h_0 de dimensiones (num_layers, batch_size, hidden_size)
            """
            batch_size, seq_len, _ = x.size()
            device = x.device

            if hx is None:
                h_init = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            else:
                h_init = hx

            h_t = [h_init[i] for i in range(self.num_layers)]

            outputs = []
            for t in range(seq_len):
                layer_input = x[:, t, :]
                for layer_idx, cell in enumerate(self.layers):
                    h_new = cell(layer_input, h_t[layer_idx])
                    h_t[layer_idx] = h_new
                    
                    if layer_idx < self.num_layers - 1:
                        layer_input = self.dropout_layer(h_new)
                    else:
                        layer_input = h_new
                
                outputs.append(h_t[-1].unsqueeze(1))

            outputs = torch.cat(outputs, dim=1)
            h_final = torch.stack(h_t, dim=0)

            return outputs, h_final


    class CustomGRURegressor(nn.Module):
        """
        Regresor final que mapea el último estado oculto de la GRU multicapa a la salida continua (fatiga física y mental).
        """
        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2, output_size: int = 2):
            super().__init__()
            self.gru = CustomGRU(input_size, hidden_size, num_layers, dropout)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Mapea la secuencia de entrada x a la estimación continua bidimensional.
            x: (batch_size, seq_len, input_size)
            Retorna: (batch_size, output_size)
            """
            out, _ = self.gru(x)
            last_step_out = out[:, -1, :]
            return self.fc(last_step_out)

else:
    class CustomGRUCell:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch no está disponible. Instale torch para usar CustomGRUCell.")

    class CustomGRU:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch no está disponible. Instale torch para usar CustomGRU.")

    class CustomGRURegressor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch no está disponible. Instale torch para usar CustomGRURegressor.")
