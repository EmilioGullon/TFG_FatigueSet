# -*- coding: utf-8 -*-
"""
Pruebas unitarias para la implementación de xLSTM/sLSTM.
"""

import torch
from fatigueset.models import CustomxLSTMRegressor, CustomxLSTM, CustomxLSTMCell


def test_xlstm_cell_shape():
    batch_size = 4
    input_size = 10
    hidden_size = 16
    
    cell = CustomxLSTMCell(input_size=input_size, hidden_size=hidden_size)
    x = torch.randn(batch_size, input_size)
    h_prev = torch.randn(batch_size, hidden_size)
    c_prev = torch.randn(batch_size, hidden_size)
    n_prev = torch.ones(batch_size, hidden_size)
    m_prev = torch.zeros(batch_size, hidden_size)
    
    h_next, c_next, n_next, m_next = cell(x, h_prev, c_prev, n_prev, m_prev)
    
    assert h_next.shape == (batch_size, hidden_size)
    assert c_next.shape == (batch_size, hidden_size)
    assert n_next.shape == (batch_size, hidden_size)
    assert m_next.shape == (batch_size, hidden_size)


def test_xlstm_layer_shape():
    batch_size = 4
    seq_len = 20
    input_size = 10
    hidden_size = 16
    num_layers = 2
    
    xlstm = CustomxLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.1)
    x = torch.randn(batch_size, seq_len, input_size)
    
    out, (h_n, c_n, n_n, m_n) = xlstm(x)
    
    assert out.shape == (batch_size, seq_len, hidden_size)
    assert h_n.shape == (num_layers, batch_size, hidden_size)
    assert c_n.shape == (num_layers, batch_size, hidden_size)
    assert n_n.shape == (num_layers, batch_size, hidden_size)
    assert m_n.shape == (num_layers, batch_size, hidden_size)


def test_xlstm_regressor_shape():
    batch_size = 4
    seq_len = 20
    input_size = 10
    hidden_size = 16
    num_layers = 2
    
    model = CustomxLSTMRegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.1,
        output_size=2
    )
    x = torch.randn(batch_size, seq_len, input_size)
    out = model(x)
    
    assert out.shape == (batch_size, 2)


def test_xlstm_backward():
    batch_size = 2
    seq_len = 5
    input_size = 4
    hidden_size = 8
    
    model = CustomxLSTMRegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        dropout=0.0,
        output_size=2
    )
    x = torch.randn(batch_size, seq_len, input_size, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None
    assert x.grad.shape == x.shape
