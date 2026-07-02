# -*- coding: utf-8 -*-
"""
Subpaquete de modelos de FatigueSet.
"""

from .rnn import FatigueSequenceDataset, RNNFatiga, train_kfold
from .lstm import CustomLSTMRegressor, CustomLSTM, CustomLSTMCell
from .gru import CustomGRURegressor, CustomGRU, CustomGRUCell
from .cnn_lstm import CustomCNNLSTMRegressor
from .tcn import CustomTCNRegressor
from .transformer import CustomTSTransformerRegressor
from .patchtst import CustomPatchTSTRegressor
from .xlstm import CustomxLSTMRegressor, CustomxLSTM, CustomxLSTMCell
from .optimizers import build_optimizer, sample_model_hyperparams
from .foundation import (
    MOMENTFatigueRegressor,
    ChronosZeroShotEvaluator,
    TimesFMZeroShotEvaluator,
    compute_crps_gaussian,
    compute_coverage,
    evaluate_probabilistic_metrics,
    finetune_moment_kfold,
    MOMENT_LOCAL,
    MOMENT_SERVER,
    CHRONOS_LOCAL,
    CHRONOS_SERVER,
)

__all__ = [
    # Dataset & RNN clásica
    'FatigueSequenceDataset',
    'RNNFatiga',
    'train_kfold',
    # Modelos propios
    'CustomLSTMRegressor', 'CustomLSTM', 'CustomLSTMCell',
    'CustomGRURegressor', 'CustomGRU', 'CustomGRUCell',
    'CustomCNNLSTMRegressor',
    'CustomTCNRegressor',
    'CustomTSTransformerRegressor',
    'CustomPatchTSTRegressor',
    'CustomxLSTMRegressor', 'CustomxLSTM', 'CustomxLSTMCell',
    # Optimizadores
    'build_optimizer', 'sample_model_hyperparams',
    # Modelos fundacionales
    'MOMENTFatigueRegressor',
    'ChronosZeroShotEvaluator',
    'TimesFMZeroShotEvaluator',
    'compute_crps_gaussian',
    'compute_coverage',
    'evaluate_probabilistic_metrics',
    'finetune_moment_kfold',
    'MOMENT_LOCAL', 'MOMENT_SERVER',
    'CHRONOS_LOCAL', 'CHRONOS_SERVER',
]
