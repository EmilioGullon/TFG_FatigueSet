# -*- coding: utf-8 -*-
"""
Pruebas unitarias para el modelo TCN.
"""

import torch
import pytest
from fatigueset.models import CustomTCNRegressor


def test_tcn_regressor_shape():
    batch_size = 4
    seq_len = 128
    input_size = 10
    num_channels = [16, 16, 16]
    kernel_size = 3
    
    model = CustomTCNRegressor(
        input_size=input_size,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=0.1,
        output_size=2
    )
    
    x = torch.randn(batch_size, seq_len, input_size)
    out = model(x)
    
    assert out.shape == (batch_size, 2)


def test_tcn_backward():
    batch_size = 2
    seq_len = 32
    input_size = 5
    num_channels = [8, 8]
    
    model = CustomTCNRegressor(
        input_size=input_size,
        num_channels=num_channels,
        kernel_size=3,
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
