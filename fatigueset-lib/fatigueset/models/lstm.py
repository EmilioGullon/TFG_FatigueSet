# -*- coding: utf-8 -*-
"""
Long Short-Term Memory (LSTM) en PyTorch implementada manualmente para FatigueSet.
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
    class CustomLSTMCell(nn.Module):
        """
        Celda LSTM manual que computa las ecuaciones clásicas de Hochreiter & Schmidhuber (1997).
        """
        def __init__(self, input_size: int, hidden_size: int):
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size

            # Parámetros para la puerta de olvido (Forget Gate)
            self.W_f = nn.Parameter(torch.empty(hidden_size, input_size))
            self.U_f = nn.Parameter(torch.empty(hidden_size, hidden_size))
            self.b_f = nn.Parameter(torch.empty(hidden_size))

            # Parámetros para la puerta de entrada (Input Gate)
            self.W_i = nn.Parameter(torch.empty(hidden_size, input_size))
            self.U_i = nn.Parameter(torch.empty(hidden_size, hidden_size))
            self.b_i = nn.Parameter(torch.empty(hidden_size))

            # Parámetros para el candidato de celda (Candidate Cell State)
            self.W_c = nn.Parameter(torch.empty(hidden_size, input_size))
            self.U_c = nn.Parameter(torch.empty(hidden_size, hidden_size))
            self.b_c = nn.Parameter(torch.empty(hidden_size))

            # Parámetros para la puerta de salida (Output Gate)
            self.W_o = nn.Parameter(torch.empty(hidden_size, input_size))
            self.U_o = nn.Parameter(torch.empty(hidden_size, hidden_size))
            self.b_o = nn.Parameter(torch.empty(hidden_size))

            self.reset_parameters()

        def reset_parameters(self):
            """Inicialización uniforme estándar (Kaiming/Uniform)."""
            stdv = 1.0 / math.sqrt(self.hidden_size)
            for p in self.parameters():
                p.data.uniform_(-stdv, stdv)

        def forward(self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Paso de tiempo individual de la celda LSTM.
            x: (batch_size, input_size)
            h_prev: (batch_size, hidden_size)
            c_prev: (batch_size, hidden_size)
            """
            f_t = torch.sigmoid(x @ self.W_f.t() + h_prev @ self.U_f.t() + self.b_f)
            i_t = torch.sigmoid(x @ self.W_i.t() + h_prev @ self.U_i.t() + self.b_i)
            c_tilde_t = torch.tanh(x @ self.W_c.t() + h_prev @ self.U_c.t() + self.b_c)
            
            c_t = f_t * c_prev + i_t * c_tilde_t
            
            o_t = torch.sigmoid(x @ self.W_o.t() + h_prev @ self.U_o.t() + self.b_o)
            h_t = o_t * torch.tanh(c_t)
            
            return h_t, c_t


    class CustomLSTM(nn.Module):
        """
        Capa LSTM multicapa que procesa secuencias completas usando CustomLSTMCell.
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
                self.layers.append(CustomLSTMCell(layer_input_size, hidden_size))

            self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        def forward(self, x: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            """
            Procesamiento de toda la secuencia temporal.
            x: (batch_size, seq_len, input_size)
            hx: tuple de (h_0, c_0) de dimensiones (num_layers, batch_size, hidden_size)
            """
            batch_size, seq_len, _ = x.size()
            device = x.device

            if hx is None:
                h_init = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
                c_init = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            else:
                h_init, c_init = hx

            h_t = [h_init[i] for i in range(self.num_layers)]
            c_t = [c_init[i] for i in range(self.num_layers)]

            outputs = []
            for t in range(seq_len):
                layer_input = x[:, t, :]
                for layer_idx, cell in enumerate(self.layers):
                    h_new, c_new = cell(layer_input, h_t[layer_idx], c_t[layer_idx])
                    h_t[layer_idx] = h_new
                    c_t[layer_idx] = c_new
                    
                    if layer_idx < self.num_layers - 1:
                        layer_input = self.dropout_layer(h_new)
                    else:
                        layer_input = h_new
                
                outputs.append(h_t[-1].unsqueeze(1))

            outputs = torch.cat(outputs, dim=1)
            h_final = torch.stack(h_t, dim=0)
            c_final = torch.stack(c_t, dim=0)

            return outputs, (h_final, c_final)


    class CustomLSTMRegressor(nn.Module):
        """
        Regresor final que mapea el último estado oculto de la LSTM multicapa a la salida continua (fatiga física y mental).
        """
        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2, output_size: int = 2):
            super().__init__()
            self.lstm = CustomLSTM(input_size, hidden_size, num_layers, dropout)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Mapea la secuencia de entrada x a la estimación continua bidimensional.
            x: (batch_size, seq_len, input_size)
            Retorna: (batch_size, output_size)
            """
            out, _ = self.lstm(x)
            last_step_out = out[:, -1, :]
            return self.fc(last_step_out)

else:
    class CustomLSTMCell:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch no está disponible. Instale torch para usar CustomLSTMCell.")

    class CustomLSTM:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch no está disponible. Instale torch para usar CustomLSTM.")

    class CustomLSTMRegressor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch no está disponible. Instale torch para usar CustomLSTMRegressor.")
