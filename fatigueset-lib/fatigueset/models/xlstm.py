# -*- coding: utf-8 -*-
"""
Extended Long Short-Term Memory (xLSTM / sLSTM) en PyTorch implementada de forma manual para FatigueSet.
Basado en: Beck, M. et al. (2024). "xLSTM: Extended Long Short-Term Memory". arXiv.
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
    class CustomxLSTMCell(nn.Module):
        """
        Celda sLSTM (Stabilized LSTM) manual que implementa la estabilización logarítmica y puertas exponenciales.
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

        def forward(
            self,
            x: torch.Tensor,
            h_prev: torch.Tensor,
            c_prev: torch.Tensor,
            n_prev: torch.Tensor,
            m_prev: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Paso individual de tiempo para la celda sLSTM.
            x: (batch_size, input_size)
            h_prev: (batch_size, hidden_size)
            c_prev: (batch_size, hidden_size)
            n_prev: (batch_size, hidden_size)
            m_prev: (batch_size, hidden_size)
            """
            # Pre-activaciones lineales
            f_tilde = x @ self.W_f.t() + h_prev @ self.U_f.t() + self.b_f
            i_tilde = x @ self.W_i.t() + h_prev @ self.U_i.t() + self.b_i
            c_tilde = torch.tanh(x @ self.W_c.t() + h_prev @ self.U_c.t() + self.b_c)
            o_tilde = x @ self.W_o.t() + h_prev @ self.U_o.t() + self.b_o

            # Estabilización en escala logarítmica para evitar desbordamiento por exponenciales
            # m_t = max(m_prev + f_tilde, i_tilde)
            m_t = torch.max(m_prev + f_tilde, i_tilde)

            # Puertas exponenciales estabilizadas
            # f_t^s = exp(m_prev + f_tilde - m_t)
            # i_t^s = exp(i_tilde - m_t)
            f_s = torch.exp(m_prev + f_tilde - m_t)
            i_s = torch.exp(i_tilde - m_t)

            # Actualización del estado de la celda y el normalizador
            c_t = f_s * c_prev + i_s * c_tilde
            n_t = f_s * n_prev + i_s

            # Estado de celda estabilizado (normalizado)
            # Para evitar división por cero en el primer paso, usamos una pequeña constante epsilon
            eps = 1e-8
            c_hat = c_t / (n_t + eps)

            # Compuerta de salida y nuevo estado oculto
            o_t = torch.sigmoid(o_tilde)
            h_t = o_t * torch.tanh(c_hat)

            return h_t, c_t, n_t, m_t


    class CustomxLSTM(nn.Module):
        """
        Capa multicapa de sLSTM (xLSTM) que procesa secuencias completas de series temporales.
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
                self.layers.append(CustomxLSTMCell(layer_input_size, hidden_size))

            self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        def forward(self, x: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
            """
            x: (batch_size, seq_len, input_size)
            hx: tuple de (h_0, c_0, n_0, m_0)
            """
            batch_size, seq_len, _ = x.size()
            device = x.device

            if hx is None:
                h_init = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
                c_init = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
                # Según el paper, el normalizador n_0 se inicializa en 1.0 y el log-normalizador m_0 en 0.0
                n_init = torch.ones(self.num_layers, batch_size, self.hidden_size, device=device)
                m_init = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            else:
                h_init, c_init, n_init, m_init = hx

            h_t = [h_init[i] for i in range(self.num_layers)]
            c_t = [c_init[i] for i in range(self.num_layers)]
            n_t = [n_init[i] for i in range(self.num_layers)]
            m_t = [m_init[i] for i in range(self.num_layers)]

            outputs = []
            for t in range(seq_len):
                layer_input = x[:, t, :]
                for layer_idx, cell in enumerate(self.layers):
                    h_new, c_new, n_new, m_new = cell(
                        layer_input,
                        h_t[layer_idx],
                        c_t[layer_idx],
                        n_t[layer_idx],
                        m_t[layer_idx]
                    )
                    h_t[layer_idx] = h_new
                    c_t[layer_idx] = c_new
                    n_t[layer_idx] = n_new
                    m_t[layer_idx] = m_new

                    if layer_idx < self.num_layers - 1:
                        layer_input = self.dropout_layer(h_new)
                    else:
                        layer_input = h_new

                outputs.append(h_t[-1].unsqueeze(1))

            outputs = torch.cat(outputs, dim=1)
            h_final = torch.stack(h_t, dim=0)
            c_final = torch.stack(c_t, dim=0)
            n_final = torch.stack(n_t, dim=0)
            m_final = torch.stack(m_t, dim=0)

            return outputs, (h_final, c_final, n_final, m_final)


    class CustomxLSTMRegressor(nn.Module):
        """
        Regresor final que mapea el último estado oculto de la sLSTM multicapa a la salida continua.
        """
        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2, output_size: int = 2):
            super().__init__()
            self.xlstm = CustomxLSTM(input_size, hidden_size, num_layers, dropout)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            x: (batch_size, seq_len, input_size)
            Retorna: (batch_size, output_size)
            """
            out, _ = self.xlstm(x)
            last_step_out = out[:, -1, :]
            return self.fc(last_step_out)

else:
    class CustomxLSTMCell:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch no está disponible. Instale torch para usar CustomxLSTMCell.")

    class CustomxLSTM:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch no está disponible. Instale torch para usar CustomxLSTM.")

    class CustomxLSTMRegressor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch no está disponible. Instale torch para usar CustomxLSTMRegressor.")
