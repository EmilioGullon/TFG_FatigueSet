# -*- coding: utf-8 -*-
"""
Pruebas unitarias para la implementación manual de LSTM.
"""

import torch
from fatigueset.models import CustomLSTMRegressor, CustomLSTM, CustomLSTMCell


def test_lstm_cell_shape():
    batch_size = 4
    input_size = 10
    hidden_size = 16
    
    cell = CustomLSTMCell(input_size=input_size, hidden_size=hidden_size)
    x = torch.randn(batch_size, input_size)
    h_prev = torch.randn(batch_size, hidden_size)
    c_prev = torch.randn(batch_size, hidden_size)
    
    h_next, c_next = cell(x, h_prev, c_prev)
    
    assert h_next.shape == (batch_size, hidden_size)
    assert c_next.shape == (batch_size, hidden_size)


def test_lstm_layer_shape():
    batch_size = 4
    seq_len = 20
    input_size = 10
    hidden_size = 16
    num_layers = 2
    
    lstm = CustomLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.1)
    x = torch.randn(batch_size, seq_len, input_size)
    
    out, (h_n, c_n) = lstm(x)
    
    assert out.shape == (batch_size, seq_len, hidden_size)
    assert h_n.shape == (num_layers, batch_size, hidden_size)
    assert c_n.shape == (num_layers, batch_size, hidden_size)


def test_lstm_regressor_shape():
    batch_size = 4
    seq_len = 20
    input_size = 10
    hidden_size = 16
    num_layers = 2
    
    model = CustomLSTMRegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.1,
        output_size=2
    )
    x = torch.randn(batch_size, seq_len, input_size)
    out = model(x)
    
    assert out.shape == (batch_size, 2)
