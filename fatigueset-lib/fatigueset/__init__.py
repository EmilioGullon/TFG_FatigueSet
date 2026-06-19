# -*- coding: utf-8 -*-
"""
FatigueSet — API pública del paquete.

Uso rápido
----------
>>> from fatigueset import FatigueSetLoader, FatigueSetProcessor
>>> from fatigueset import FatigueSetValidator, FeatureExtractor
>>> from fatigueset import utils
"""

from .loader     import (DataLoader, FatigueSetLoader, convertir_timestamp,
                          PARTICIPANTES, SESIONES, FASES_MEDICION,
                          INTENSIDAD_NUM, CANALES_EEG, ARCHIVOS_POR_SENSOR)
from .processor  import DataProcessor, FatigueSetProcessor
from .pipeline   import FatigueSetPipeline
from .validators import DataValidator, FatigueSetValidator
from .features   import FeatureExtractor
from .           import utils
from .models.rnn import (FatigueSequenceDataset, RNNFatiga, train_kfold)

__version__ = '0.1.0'

__all__ = [
    # Clases principales
    'DataLoader',
    'FatigueSetLoader',
    'DataProcessor',
    'FatigueSetProcessor',
    'FatigueSetPipeline',
    'DataValidator',
    'FatigueSetValidator',
    'FeatureExtractor',
    'FatigueSequenceDataset',
    'RNNFatiga',
    'train_kfold',
    # Módulo de visualización
    'utils',
    # Función de utilidad
    'convertir_timestamp',
    # Constantes
    'PARTICIPANTES',
    'SESIONES',
    'FASES_MEDICION',
    'INTENSIDAD_NUM',
    'CANALES_EEG',
    'ARCHIVOS_POR_SENSOR',
]