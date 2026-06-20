# -*- coding: utf-8 -*-
"""
Pruebas unitarias para el modelo híbrido CNN-LSTM.
"""

import torch
from fatigueset.models import CustomCNNLSTMRegressor


def test_cnn_lstm_regressor_shape():
    batch_size = 4
    seq_len = 128
    input_size = 10
    conv_channels = 32
    hidden_size = 16
    num_layers = 1
    
    model = CustomCNNLSTMRegressor(
        input_size=input_size,
        conv_channels=conv_channels,
        kernel_size=3,
        pool_size=2,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.0,
        output_size=2
    )
    
    x = torch.randn(batch_size, seq_len, input_size)
    out = model(x)
    
    assert out.shape == (batch_size, 2)


def test_cnn_lstm_backward():
    batch_size = 2
    seq_len = 32
    input_size = 5
    
    model = CustomCNNLSTMRegressor(
        input_size=input_size,
        conv_channels=8,
        kernel_size=3,
        pool_size=2,
        hidden_size=8,
        num_layers=1,
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
