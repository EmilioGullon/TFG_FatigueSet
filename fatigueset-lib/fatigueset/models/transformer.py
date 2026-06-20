# -*- coding: utf-8 -*-
"""
Time Series Transformer en PyTorch implementado de forma manual para FatigueSet.
"""

from __future__ import annotations
import math
from typing import Optional

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


if nn is not None:
    class PositionalEncoding(nn.Module):
        """
        Codificación Posicional Sinusoidal manual para mantener la noción
        del orden temporal en el modelo Transformer.
        """
        def __init__(self, d_model: int, max_len: int = 1000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            
            # Ajuste de frecuencias exponenciales para sin y cos
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            
            # Asignación a canales pares e impares
            pe[:, 0::2] = torch.sin(position * div_term[:pe[:, 0::2].size(1)])
            if d_model > 1:
                pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
                
            pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
            self.register_buffer('pe', pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (B, L, d_model)
            return x + self.pe[:, :x.size(1)]


    class CustomMultiHeadAttention(nn.Module):
        """
        Mecanismo de Auto-Atención Multi-Cabeza (Multi-Head Self-Attention)
        implementado manualmente desde cero.
        """
        def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
            super().__init__()
            assert d_model % num_heads == 0, "d_model debe ser divisible por num_heads"
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            
            # Proyecciones lineales para Query, Key, Value
            self.q_linear = nn.Linear(d_model, d_model)
            self.k_linear = nn.Linear(d_model, d_model)
            self.v_linear = nn.Linear(d_model, d_model)
            
            # Proyección lineal final de salida
            self.out_linear = nn.Linear(d_model, d_model)
            
            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            # Entrada: (B, L, d_model)
            batch_size = q.size(0)
            seq_len = q.size(1)
            
            # 1. Proyecciones lineales y reformateo para procesar multi-cabeza en paralelo
            # De (B, L, d_model) a (B, H, L, d_k)
            queries = self.q_linear(q).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            keys = self.k_linear(k).view(batch_size, k.size(1), self.num_heads, self.d_k).transpose(1, 2)
            values = self.v_linear(v).view(batch_size, v.size(1), self.num_heads, self.d_k).transpose(1, 2)
            
            # 2. Scaled Dot-Product Attention
            # scores shape: (B, H, L_q, L_k)
            scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
                
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # context shape: (B, H, L_q, d_k)
            context = torch.matmul(attn_weights, values)
            
            # 3. Concatenar cabezas y proyectar salida
            # De (B, H, L_q, d_k) a (B, L_q, d_model)
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            
            return self.out_linear(context)


    class CustomTransformerEncoderLayer(nn.Module):
        """
        Capa individual de Codificador Transformer. Combina auto-atención multi-cabeza,
        conexiones residuales, normalización de capa y una red Feed-Forward.
        """
        def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1):
            super().__init__()
            self.self_attn = CustomMultiHeadAttention(d_model, num_heads, dropout)
            
            # Feed-Forward Network (FFN)
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            
            # Normalizaciones y regularizaciones
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            # Subcapa 1: Auto-Atención + Add & Norm
            attn_out = self.self_attn(x, x, x, mask)
            x = self.norm1(x + self.dropout1(attn_out))
            
            # Subcapa 2: FFN + Add & Norm
            ffn_out = self.linear2(self.dropout(self.relu(self.linear1(x))))
            x = self.norm2(x + self.dropout2(ffn_out))
            
            return x


    class CustomTSTransformerRegressor(nn.Module):
        """
        Regresor completo basado en Transformer para series temporales.
        Proyecta características, agrega posición, pasa por el stack de encoders,
        aplica Global Average Pooling temporal y proyecta a la salida de fatiga bidimensional (2,).
        """
        def __init__(
            self,
            input_size: int,
            d_model: int = 64,
            num_heads: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 128,
            dropout: float = 0.1,
            output_size: int = 2
        ):
            super().__init__()
            self.input_size = input_size
            self.d_model = d_model
            self.num_heads = num_heads
            self.num_layers = num_layers
            self.dim_feedforward = dim_feedforward
            self.dropout = dropout
            self.output_size = output_size
            
            # Proyección lineal inicial
            self.input_projection = nn.Linear(input_size, d_model)
            
            # Codificador posicional
            self.pos_encoder = PositionalEncoding(d_model)
            
            # Stack de codificación del Transformer
            self.encoder_layers = nn.ModuleList([
                CustomTransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout
                ) for _ in range(num_layers)
            ])
            
            # Capa lineal final de regresión
            self.fc = nn.Linear(d_model, output_size)

        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            # x shape: (batch_size, seq_len, input_size)
            
            # 1. Mapear canales de sensores a d_model
            out = self.input_projection(x)
            
            # 2. Añadir información posicional temporal
            out = self.pos_encoder(out)
            
            # 3. Aplicar capas del encoder
            for layer in self.encoder_layers:
                out = layer(out, mask)
                
            # 4. Global Average Pooling sobre el eje de tiempo (secuencia)
            pooled = out.mean(dim=1)
            
            # 5. Predicción bidimensional final (física + mental)
            return self.fc(pooled)

else:
    class CustomTSTransformerRegressor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch no está disponible. Instale torch para usar CustomTSTransformerRegressor.")
