# -*- coding: utf-8 -*-
"""
Extended Long Short-Term Memory (xLSTM) - Variante sLSTM en PyTorch implementada manualmente para FatigueSet.
Referencia: Beck, M. et al. (2024). "xLSTM: Extended Long Short-Term Memory". arXiv preprint arXiv:2405.04517.
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
    class CustomsLSTMCell(nn.Module):
        """
        Celda sLSTM manual (Stabilized LSTM) con puertas exponenciales.
        Utiliza m_t y n_t para la estabilización numérica frente al crecimiento exponencial.
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

            # Parámetros para la puerta de salida (Output Gate)
            self.W_o = nn.Parameter(torch.empty(hidden_size, input_size))
            self.U_o = nn.Parameter(torch.empty(hidden_size, hidden_size))
            self.b_o = nn.Parameter(torch.empty(hidden_size))

            # Parámetros para la entrada de celda (Cell Input Candidate)
            self.W_z = nn.Parameter(torch.empty(hidden_size, input_size))
            self.U_z = nn.Parameter(torch.empty(hidden_size, hidden_size))
            self.b_z = nn.Parameter(torch.empty(hidden_size))

            self.reset_parameters()

        def reset_parameters(self):
            """Inicialización uniforme estándar (Kaiming/Uniform)."""
            stdv = 1.0 / math.sqrt(self.hidden_size)
            for p in self.parameters():
                p.data.uniform_(-stdv, stdv)

        def forward(self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor, n_prev: torch.Tensor, m_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Paso de tiempo individual de la celda sLSTM.
            x: (batch_size, input_size)
            h_prev: (batch_size, hidden_size)
            c_prev: (batch_size, hidden_size)
            n_prev: (batch_size, hidden_size)
            m_prev: (batch_size, hidden_size)
            """
            # Pre-activaciones
            f_tilde = x @ self.W_f.t() + h_prev @ self.U_f.t() + self.b_f
            i_tilde = x @ self.W_i.t() + h_prev @ self.U_i.t() + self.b_i
            o_tilde = x @ self.W_o.t() + h_prev @ self.U_o.t() + self.b_o
            z_tilde = x @ self.W_z.t() + h_prev @ self.U_z.t() + self.b_z

            # 1. Actualización del estabilizador (m_t)
            # m_t = max(m_{t-1} + f_tilde, i_tilde)
            m_t = torch.max(m_prev + f_tilde, i_tilde)

            # 2. Exponenciación de puertas con estabilización
            # f_t = exp(f_tilde + m_prev - m_t)
            # i_t = exp(i_tilde - m_t)
            f_t = torch.exp(f_tilde + m_prev - m_t)
            i_t = torch.exp(i_tilde - m_t)

            # 3. Puerta de salida (sigmoide normal)
            o_t = torch.sigmoid(o_tilde)

            # 4. Candidato de celda (tanh)
            z_t = torch.tanh(z_tilde)

            # 5. Estado de celda y normalizador
            c_t = f_t * c_prev + i_t * z_t
            n_t = f_t * n_prev + i_t

            # 6. Salida oculta estabilizada
            # Añadimos epsilon = 1e-8 para evitar división por cero en la normalización
            h_t = o_t * (c_t / (n_t + 1e-8))

            return h_t, c_t, n_t, m_t


    class CustomsLSTM(nn.Module):
        """
        Capa sLSTM multicapa que procesa secuencias completas usando CustomsLSTMCell.
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
                self.layers.append(CustomsLSTMCell(layer_input_size, hidden_size))

            self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        def forward(self, x: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
            """
            Procesamiento de toda la secuencia temporal.
            x: (batch_size, seq_len, input_size)
            hx: tuple de (h_0, c_0, n_0, m_0) de dimensiones (num_layers, batch_size, hidden_size)
            """
            batch_size, seq_len, _ = x.size()
            device = x.device

            if hx is None:
                h_init = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
                c_init = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
                # n_0 se inicializa a unos para la normalización inicial
                n_init = torch.ones(self.num_layers, batch_size, self.hidden_size, device=device)
                # m_0 se inicializa a ceros como log-space inicial
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
        Regresor final xLSTM (sLSTM) para predecir fatiga física y mental.
        """
        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2, output_size: int = 2):
            super().__init__()
            self.lstm = CustomsLSTM(input_size, hidden_size, num_layers, dropout)
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
    class CustomsLSTMCell:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch no está disponible. Instale torch para usar CustomsLSTMCell.")

    class CustomsLSTM:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch no está disponible. Instale torch para usar CustomsLSTM.")

    class CustomxLSTMRegressor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch no está disponible. Instale torch para usar CustomxLSTMRegressor.")
