# -*- coding: utf-8 -*-
"""
Pruebas unitarias para el modelo Transformer.
"""

import torch
import pytest
from fatigueset.models import CustomTSTransformerRegressor
from fatigueset.models.transformer import PositionalEncoding, CustomMultiHeadAttention, CustomTransformerEncoderLayer


def test_positional_encoding_shape():
    max_len = 100
    d_model = 32
    pe = PositionalEncoding(d_model=d_model, max_len=max_len)
    
    x = torch.randn(4, 50, d_model)
    out = pe(x)
    
    assert out.shape == x.shape


def test_multi_head_attention_shape():
    batch_size = 4
    seq_len = 32
    d_model = 32
    num_heads = 4
    
    mha = CustomMultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    
    x = torch.randn(batch_size, seq_len, d_model)
    out = mha(x, x, x)
    
    assert out.shape == (batch_size, seq_len, d_model)


def test_transformer_encoder_layer_shape():
    batch_size = 4
    seq_len = 32
    d_model = 32
    num_heads = 4
    dim_feedforward = 64
    
    layer = CustomTransformerEncoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=0.1
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    out = layer(x)
    
    assert out.shape == (batch_size, seq_len, d_model)


def test_transformer_regressor_shape():
    batch_size = 4
    seq_len = 64
    input_size = 10
    d_model = 16
    num_heads = 2
    
    model = CustomTSTransformerRegressor(
        input_size=input_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.1,
        output_size=2
    )
    
    x = torch.randn(batch_size, seq_len, input_size)
    out = model(x)
    
    assert out.shape == (batch_size, 2)


def test_transformer_mask_support():
    batch_size = 2
    seq_len = 8
    input_size = 5
    d_model = 8
    num_heads = 2
    
    model = CustomTSTransformerRegressor(
        input_size=input_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        output_size=2
    )
    
    x = torch.randn(batch_size, seq_len, input_size)
    # Máscara binaria de forma (batch_size, 1, seq_len, seq_len)
    mask = torch.ones(batch_size, 1, seq_len, seq_len)
    mask[:, :, :, -2:] = 0  # Enmascarar las últimas dos posiciones temporales
    
    out = model(x, mask=mask)
    assert out.shape == (batch_size, 2)


def test_transformer_backward():
    batch_size = 2
    seq_len = 16
    input_size = 5
    d_model = 8
    num_heads = 2
    
    model = CustomTSTransformerRegressor(
        input_size=input_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        output_size=2
    )
    
    x = torch.randn(batch_size, seq_len, input_size)
    targets = torch.randn(batch_size, 2)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    
    preds = model(x)
    loss = loss_fn(preds, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Si llega aquí sin lanzar errores, el paso backward es estable y correcto
    assert True
