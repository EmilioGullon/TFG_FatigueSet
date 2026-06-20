# -*- coding: utf-8 -*-
"""
Pruebas unitarias para el modelo PatchTST.
"""

import torch
import pytest
from fatigueset.models import CustomPatchTSTRegressor


def test_patchtst_regressor_shape():
    batch_size = 4
    seq_len = 128
    input_size = 10
    d_model = 16
    num_heads = 2
    
    model = CustomPatchTSTRegressor(
        input_size=input_size,
        patch_len=16,
        stride=8,
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


def test_patchtst_backward():
    batch_size = 2
    seq_len = 64
    input_size = 5
    d_model = 8
    num_heads = 2
    
    model = CustomPatchTSTRegressor(
        input_size=input_size,
        patch_len=8,
        stride=4,
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
