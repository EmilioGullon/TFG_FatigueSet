# -*- coding: utf-8 -*-
"""
PatchTST (Patch Time Series Transformer) en PyTorch implementado de forma manual para FatigueSet.
"""

from __future__ import annotations
import math
from typing import Optional

try:
    import torch
    import torch.nn as nn
    from .transformer import CustomTransformerEncoderLayer
except ImportError:
    torch = None
    nn = None
    CustomTransformerEncoderLayer = None


if nn is not None:
    class CustomPatchTSTRegressor(nn.Module):
        """
        Regresor basado en la arquitectura PatchTST (Nie et al., 2022).
        Divide la señal temporal univariada de cada canal en parches de tamaño P con stride S,
        aplica independecia de canales compartiendo los pesos del codificador Transformer,
        y finalmente realiza la regresión bidimensional (fatiga física + mental).
        """
        def __init__(
            self,
            input_size: int,         # Número de canales de entrada (M)
            patch_len: int = 16,     # Tamaño del parche (P)
            stride: int = 8,         # Paso de los parches (S)
            d_model: int = 64,       # Dimensión del modelo Transformer
            num_heads: int = 4,      # Número de cabezas de auto-atención
            num_layers: int = 2,     # Número de capas del codificador
            dim_feedforward: int = 128,  # Dimensión oculta de la red feed-forward
            dropout: float = 0.1,    # Probabilidad de Dropout
            output_size: int = 2     # Tamaño de salida
        ):
            super().__init__()
            self.input_size = input_size
            self.patch_len = patch_len
            self.stride = stride
            self.d_model = d_model
            self.num_heads = num_heads
            self.num_layers = num_layers
            self.dim_feedforward = dim_feedforward
            self.dropout = dropout
            self.output_size = output_size
            
            # Proyección lineal para mapear el tamaño de parche P a d_model
            self.linear_projection = nn.Linear(patch_len, d_model)
            
            # Codificación posicional aprendible.
            # max_patches es el número máximo de parches que soportamos estáticamente.
            self.max_patches = 100
            self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_patches, d_model))
            nn.init.normal_(self.pos_embedding, std=0.02)
            
            self.dropout_layer = nn.Dropout(dropout)
            
            # Stack de capas codificadoras del Transformer reutilizando CustomTransformerEncoderLayer
            self.encoder_layers = nn.ModuleList([
                CustomTransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout
                ) for _ in range(num_layers)
            ])
            
            # Capa de salida lineal para regresión fisiológica multicanal
            # Proyecta las características agregadas de todos los canales a la predicción final (2,)
            self.fc = nn.Linear(input_size * d_model, output_size)

        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            # x shape: (batch_size, seq_len, input_size)
            batch_size, seq_len, input_size = x.shape
            assert input_size == self.input_size, f"Input size {input_size} no coincide con el configurado {self.input_size}"
            
            # 1. Independencia de canales (Channel Independence)
            # Reorganizar a (B, M, L)
            x_trans = x.transpose(1, 2)
            # Aplanar Batch y Canales para procesarlos como series univariadas independientes
            # x_ind shape: (batch_size * input_size, seq_len)
            x_ind = x_trans.reshape(batch_size * input_size, seq_len)
            
            # 2. Segmentación en parches (Patching)
            # Extraer parches locales a lo largo de la dimensión temporal (dimension=-1)
            # x_patched shape: (batch_size * input_size, num_patches, patch_len)
            x_patched = x_ind.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            num_patches = x_patched.size(1)
            
            assert num_patches <= self.max_patches, f"Secuencia demasiado larga. Número de parches {num_patches} supera el máximo de {self.max_patches}"
            
            # 3. Proyección e inyección posicional
            # Proyectar parches a d_model
            out = self.linear_projection(x_patched) # shape: (batch_size * input_size, num_patches, d_model)
            
            # Sumar codificación posicional aprendible
            out = out + self.pos_embedding[:, :num_patches, :]
            out = self.dropout_layer(out)
            
            # 4. Procesar a través del codificador Transformer
            for layer in self.encoder_layers:
                out = layer(out, mask)
                
            # 5. Pooling y proyección final
            # Global Average Pooling a lo largo de los parches
            pooled = out.mean(dim=1) # shape: (batch_size * input_size, d_model)
            
            # Separar canales e inyectar el batch original
            # shape: (batch_size, input_size, d_model)
            pooled = pooled.view(batch_size, input_size, self.d_model)
            
            # Aplanar canales para la predicción multivariada final
            # shape: (batch_size, input_size * d_model)
            flat = pooled.view(batch_size, input_size * self.d_model)
            
            # Regresión final
            return self.fc(flat)

else:
    class CustomPatchTSTRegressor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch no está disponible. Instale torch para usar CustomPatchTSTRegressor.")
