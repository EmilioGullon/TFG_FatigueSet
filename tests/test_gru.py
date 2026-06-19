# -*- coding: utf-8 -*-
"""
Pruebas unitarias para la implementación manual de GRU.
"""

import torch
from fatigueset.models import CustomGRURegressor, CustomGRU, CustomGRUCell


def test_gru_cell_shape():
    batch_size = 4
    input_size = 10
    hidden_size = 16
    
    cell = CustomGRUCell(input_size=input_size, hidden_size=hidden_size)
    x = torch.randn(batch_size, input_size)
    h_prev = torch.randn(batch_size, hidden_size)
    
    h_next = cell(x, h_prev)
    
    assert h_next.shape == (batch_size, hidden_size)


def test_gru_layer_shape():
    batch_size = 4
    seq_len = 20
    input_size = 10
    hidden_size = 16
    num_layers = 2
    
    gru = CustomGRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.1)
    x = torch.randn(batch_size, seq_len, input_size)
    
    out, h_n = gru(x)
    
    assert out.shape == (batch_size, seq_len, hidden_size)
    assert h_n.shape == (num_layers, batch_size, hidden_size)


def test_gru_regressor_shape():
    batch_size = 4
    seq_len = 20
    input_size = 10
    hidden_size = 16
    num_layers = 2
    
    model = CustomGRURegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.1,
        output_size=2
    )
    x = torch.randn(batch_size, seq_len, input_size)
    out = model(x)
    
    assert out.shape == (batch_size, 2)
