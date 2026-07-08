    # -*- coding: utf-8 -*-
"""
Módulo de configuración y construcción de optimizadores para FatigueSet.

Este módulo provee:
  - Espacios de búsqueda de hiperparámetros de optimizadores compatibles con Optuna.
  - Espacios de búsqueda de arquitectura por familia de modelos (según el estudio teórico).
  - Una función de fábrica ``build_optimizer`` para instanciar el optimizador seleccionado
    a partir del diccionario de hiperparámetros sugerido por Optuna.
  - Funciones ``sample_*`` que devuelven el espacio de búsqueda de cada arquitectura.
  - Una función ``sample_model_hyperparams`` unificada para la función objetivo de Optuna.

Referencia:
  Optimización de Modelos en Optuna (TFG, 2026).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Construcción del Optimizador
# ---------------------------------------------------------------------------

def build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    alpha: float = 0.99,
) -> torch.optim.Optimizer:
    """
    Instancia un optimizador de PyTorch a partir de su nombre y parámetros.

    Los optimizadores soportados son: ``Adam``, ``AdamW``, ``RMSprop`` y ``SGD``.

    Parámetros
    ----------
    model : nn.Module
        El modelo cuyos parámetros se van a optimizar.
    optimizer_name : str
        Nombre del optimizador. Debe ser uno de ``['Adam', 'AdamW', 'RMSprop', 'SGD']``.
    lr : float
        Tasa de aprendizaje (learning rate).
    weight_decay : float, opcional
        Término de regularización L2 (penalización de norma al cuadrado). Por defecto 0.0.
        En ``AdamW``, este parámetro desacopla el decaimiento de pesos de la actualización
        basada en momentos (Loshchilov & Hutter, 2017).
    momentum : float, opcional
        Coeficiente de momentum para SGD. Por defecto 0.9.
        Solo utilizado cuando ``optimizer_name == 'SGD'``.
    alpha : float, opcional
        Factor de suavizado para la media móvil cuadrática de RMSprop. Por defecto 0.99.
        Solo utilizado cuando ``optimizer_name == 'RMSprop'``.

    Retorna
    -------
    torch.optim.Optimizer
        Instancia del optimizador configurado.

    Raises
    ------
    ValueError
        Si ``optimizer_name`` no corresponde a ningún optimizador soportado.

    Ejemplos
    --------
    >>> opt = build_optimizer(model, 'AdamW', lr=1e-3, weight_decay=1e-4)
    >>> opt = build_optimizer(model, 'SGD', lr=1e-2, momentum=0.9, weight_decay=1e-4)
    """
    params = model.parameters()
    name = optimizer_name.strip()

    if name == "Adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "AdamW":
        # AdamW desacopla el decaimiento de pesos, siendo preferible para regularización
        # (Loshchilov & Hutter, 2017: "Decoupled Weight Decay Regularization").
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == "RMSprop":
        return torch.optim.RMSprop(
            params, lr=lr, alpha=alpha, weight_decay=weight_decay
        )
    elif name == "SGD":
        return torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True
        )
    else:
        raise ValueError(
            f"Optimizador '{optimizer_name}' no reconocido. "
            f"Opciones válidas: ['Adam', 'AdamW', 'RMSprop', 'SGD']."
        )


# ---------------------------------------------------------------------------
# 2. Espacio de búsqueda de hiperparámetros de optimizador en Optuna
# ---------------------------------------------------------------------------

def sample_optimizer_params(trial: Any, prefix: str) -> Dict[str, Any]:
    """
    Define el espacio de búsqueda de hiperparámetros del optimizador para Optuna.

    Utiliza búsqueda condicional: primero se selecciona el optimizador, y luego
    se activan únicamente los hiperparámetros relevantes para ese algoritmo.
    Esto evita sugerir parámetros irrelevantes (p. ej. momentum para Adam),
    lo que mejoraría la eficiencia de la búsqueda bayesiana al reducir la
    dimensionalidad del espacio de búsqueda (Tree-structured Parzen Estimator).

    Parámetros
    ----------
    trial : optuna.trial.Trial
        Objeto de ensayo de Optuna.
    prefix : str
        Prefijo para los nombres de los hiperparámetros (evita colisiones entre modelos).

    Retorna
    -------
    Dict[str, Any]
        Diccionario con las claves: ``optimizer``, ``lr``, ``weight_decay`` y,
        condicionalmente, ``momentum`` (SGD) o ``alpha`` (RMSprop).
    """
    optimizer_name = trial.suggest_categorical(
        f"{prefix}_optimizer", ["Adam", "AdamW", "RMSprop", "SGD"]
    )
    lr = trial.suggest_float(f"{prefix}_lr", 1e-5, 1e-2, log=True)

    # weight_decay es aplicable a todos los optimizadores
    weight_decay = trial.suggest_float(f"{prefix}_weight_decay", 1e-7, 1e-3, log=True)

    params: Dict[str, Any] = {
        "optimizer": optimizer_name,
        "lr": lr,
        "weight_decay": weight_decay,
    }

    if optimizer_name == "SGD":
        # El momentum de Nesterov es clave en SGD para la convergencia en deep learning
        params["momentum"] = trial.suggest_float(f"{prefix}_momentum", 0.7, 0.99)
    elif optimizer_name == "RMSprop":
        # El factor alpha controla el suavizado exponencial de la media cuadrática
        params["alpha"] = trial.suggest_float(f"{prefix}_alpha", 0.9, 0.999)

    return params


# ---------------------------------------------------------------------------
# 3. Espacios de búsqueda de arquitectura por familia de modelos
# ---------------------------------------------------------------------------

def sample_random_forest(trial: Any) -> Dict[str, Any]:
    """
    Espacio de búsqueda de hiperparámetros para Random Forest (sklearn).

    El ``n_estimators`` limita intencionalmente el máximo a 300 para equilibrar
    el tiempo de búsqueda con la robustez del ensamble. El parámetro ``max_features``
    controla la diversidad de los árboles y es clave para evitar correlaciones entre ellos.

    Parámetros
    ----------
    trial : optuna.trial.Trial

    Retorna
    -------
    Dict[str, Any]
    """
    return {
        "model_type": "RandomForest",
        "n_estimators": trial.suggest_int("rf_n_estimators", 50, 300, step=50),
        "max_depth": trial.suggest_categorical("rf_max_depth", [None, 5, 10, 20, 30]),
        "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
            "rf_max_features", ["sqrt", "log2", 0.5, 0.7]
        ),
    }


def sample_lstm(trial: Any) -> Dict[str, Any]:
    """
    Espacio de búsqueda de hiperparámetros para LSTM personalizada (``CustomLSTMRegressor``).

    El parámetro ``optimizer`` se incluye en el espacio de búsqueda de arquitectura
    para LSTM dado que el tipo de optimizador puede tener un impacto significativo
    en la estabilidad y convergencia de redes recurrentes (Goodfellow et al., 2016).

    Parámetros
    ----------
    trial : optuna.trial.Trial

    Retorna
    -------
    Dict[str, Any]
    """
    opt_params = sample_optimizer_params(trial, prefix="lstm")
    return {
        "model_type": "LSTM",
        "hidden_size": trial.suggest_int("lstm_hidden_size", 16, 256, step=16),
        "num_layers": trial.suggest_int("lstm_num_layers", 1, 4),
        "dropout": trial.suggest_float("lstm_dropout", 0.0, 0.5),
        "batch_size": trial.suggest_categorical(
            "lstm_batch_size", [16, 32, 64, 128]
        ),
        **opt_params,
    }


def sample_gru(trial: Any) -> Dict[str, Any]:
    """
    Espacio de búsqueda de hiperparámetros para GRU personalizada (``CustomGRURegressor``).

    La GRU utiliza un espacio más compacto que la LSTM al eliminar el canal de memoria
    explícito. La regularización ``weight_decay`` es especialmente relevante dado que
    la GRU puede sobreajustarse con mayor rapidez en conjuntos de datos pequeños.

    Parámetros
    ----------
    trial : optuna.trial.Trial

    Retorna
    -------
    Dict[str, Any]
    """
    opt_params = sample_optimizer_params(trial, prefix="gru")
    return {
        "model_type": "GRU",
        "hidden_size": trial.suggest_int("gru_hidden_size", 16, 256, step=16),
        "num_layers": trial.suggest_int("gru_num_layers", 1, 4),
        "dropout": trial.suggest_float("gru_dropout", 0.0, 0.5),
        "batch_size": trial.suggest_categorical(
            "gru_batch_size", [16, 32, 64, 128]
        ),
        **opt_params,
    }


def sample_cnn_lstm(trial: Any) -> Dict[str, Any]:
    """
    Espacio de búsqueda de hiperparámetros para CNN-LSTM híbrida (``CustomCNNLSTMRegressor``).

    Los hiperparámetros convolucionales (``conv_channels``, ``kernel_size``, ``pool_size``)
    gobiernan la extracción de características locales, mientras que los recurrentes
    (``hidden_size``, ``num_layers``) controlan la capacidad de modelado secuencial a largo plazo.

    Parámetros
    ----------
    trial : optuna.trial.Trial

    Retorna
    -------
    Dict[str, Any]
    """
    opt_params = sample_optimizer_params(trial, prefix="cnn_lstm")
    return {
        "model_type": "CNN-LSTM",
        "conv_channels": trial.suggest_int("cnn_conv_channels", 16, 128, step=16),
        "kernel_size": trial.suggest_int("cnn_kernel_size", 2, 8),
        "pool_size": trial.suggest_int("cnn_pool_size", 2, 4),
        "hidden_size": trial.suggest_int("cnn_lstm_hidden_size", 32, 256, step=32),
        "num_layers": trial.suggest_int("cnn_lstm_layers", 1, 3),
        "dropout": trial.suggest_float("cnn_dropout", 0.1, 0.5),
        "batch_size": trial.suggest_categorical(
            "cnn_batch_size", [16, 32, 64, 128]
        ),
        **opt_params,
    }


def sample_tcn(trial: Any) -> Dict[str, Any]:
    """
    Espacio de búsqueda de hiperparámetros para TCN (``CustomTCNRegressor``).

    Los ``num_channels`` se parametrizan como índice categórico para evitar que Optuna
    sugiera listas de longitud variable (incompatibles con el protocolo de sugerencia
    escalar de TPE). El ``kernel_size`` controla el campo receptivo base y la profundidad
    de la red determina cuántas dilaciones exponenciales se aplican.

    Parámetros
    ----------
    trial : optuna.trial.Trial

    Retorna
    -------
    Dict[str, Any]
    """
    # Configuraciones predefinidas de canales para búsqueda categórica
    channels_options = [
        [16, 32, 64],
        [32, 64, 128],
        [64, 128, 256],
        [32, 64, 64, 128],
        [64, 64, 128, 128],
    ]
    channels_idx = trial.suggest_categorical("tcn_channels_idx", list(range(len(channels_options))))
    num_channels = channels_options[channels_idx]

    opt_params = sample_optimizer_params(trial, prefix="tcn")
    return {
        "model_type": "TCN",
        "num_channels": num_channels,
        "kernel_size": trial.suggest_int("tcn_kernel_size", 2, 8),
        "dropout": trial.suggest_float("tcn_dropout", 0.0, 0.5),
        "batch_size": trial.suggest_categorical(
            "tcn_batch_size", [16, 32, 64, 128]
        ),
        **opt_params,
    }


def sample_transformer(trial: Any) -> Dict[str, Any]:
    """
    Espacio de búsqueda de hiperparámetros para el Transformer de series temporales
    (``CustomTSTransformerRegressor``).

    La restricción de divisibilidad ``d_model % nhead == 0`` se garantiza algebraicamente
    construyendo ``d_model = nhead * d_model_multiplier``, lo que permite una búsqueda
    bayesiana libre sin riesgo de errores en tiempo de ejecución por incompatibilidad
    de dimensiones en la capa de auto-atención multi-cabeza.

    Parámetros
    ----------
    trial : optuna.trial.Trial

    Retorna
    -------
    Dict[str, Any]
    """
    nhead = trial.suggest_categorical("trans_nhead", [2, 4, 8])
    d_model_multiplier = trial.suggest_int("trans_d_model_multiplier", 8, 32)
    d_model = nhead * d_model_multiplier  # Garantiza divisibilidad

    opt_params = sample_optimizer_params(trial, prefix="trans")
    return {
        "model_type": "Transformer",
        "d_model": d_model,
        "num_heads": nhead,
        "num_layers": trial.suggest_int("trans_num_layers", 1, 6),
        "dim_feedforward": trial.suggest_int("trans_dim_feedforward", 64, 512, step=64),
        "dropout": trial.suggest_float("trans_dropout", 0.0, 0.4),
        "batch_size": trial.suggest_categorical(
            "trans_batch_size", [16, 32, 64, 128]
        ),
        **opt_params,
    }


def sample_patchtst(trial: Any) -> Dict[str, Any]:
    """
    Espacio de búsqueda de hiperparámetros para PatchTST (``CustomPatchTSTRegressor``).

    El parcheo de la serie (``patch_len`` y ``stride``) determina cuántos tokens se
    generan a partir de la secuencia de entrada, afectando directamente la complejidad
    de la auto-atención cuadrática. Se utiliza búsqueda condicional por coherencia:
    ``n_heads`` debe dividir a ``d_model`` (que se fija como múltiplo de n_heads).

    Parámetros
    ----------
    trial : optuna.trial.Trial

    Retorna
    -------
    Dict[str, Any]
    """
    n_heads = trial.suggest_categorical("patch_n_heads", [4, 8])
    d_model_multiplier = trial.suggest_int("patch_d_model_mult", 8, 32)
    d_model = n_heads * d_model_multiplier  # Garantiza divisibilidad

    opt_params = sample_optimizer_params(trial, prefix="patch")
    return {
        "model_type": "PatchTST",
        "patch_len": trial.suggest_int("patch_len", 4, 32, step=4),
        "stride": trial.suggest_int("patch_stride", 2, 16, step=2),
        "d_model": d_model,
        "num_heads": n_heads,
        "num_layers": trial.suggest_int("patch_encoder_layers", 2, 6),
        "dim_feedforward": trial.suggest_int("patch_dim_ff", 64, 256, step=64),
        "dropout": trial.suggest_float("patch_dropout", 0.0, 0.4),
        "batch_size": trial.suggest_categorical(
            "patch_batch_size", [16, 32, 64]
        ),
        **opt_params,
    }


def sample_xlstm(trial: Any) -> Dict[str, Any]:
    """
    Espacio de búsqueda de hiperparámetros para xLSTM/sLSTM (``CustomxLSTMRegressor``).

    La arquitectura xLSTM implementada en FatigueSet corresponde a la variante sLSTM
    (memoria escalar estabilizada). El parámetro ``conv1d_kernel_size`` controla
    el tamaño del campo receptivo de la capa convolucional 1D interna.

    Parámetros
    ----------
    trial : optuna.trial.Trial

    Retorna
    -------
    Dict[str, Any]
    """
    opt_params = sample_optimizer_params(trial, prefix="xlstm")
    return {
        "model_type": "xLSTM",
        "hidden_size": trial.suggest_int("xlstm_hidden_size", 32, 256, step=32),
        "num_layers": trial.suggest_int("xlstm_num_layers", 1, 4),
        "dropout": trial.suggest_float("xlstm_dropout", 0.0, 0.4),
        "batch_size": trial.suggest_categorical(
            "xlstm_batch_size", [16, 32, 64]
        ),
        **opt_params,
    }


# ---------------------------------------------------------------------------
# 4. Función unificada de muestreo de hiperparámetros
# ---------------------------------------------------------------------------

def sample_model_hyperparams(trial: Any, model_family: Optional[str] = None) -> Dict[str, Any]:
    """
    Función de muestreo unificada para la función objetivo de Optuna.

    Si ``model_family`` es ``None``, la familia de modelos se selecciona dinámicamente
    como hiperparámetro categórico, permitiendo la búsqueda jerárquica y condicional
    entre familias de modelos en un único estudio de Optuna.

    Si ``model_family`` se especifica (p. ej. ``'LSTM'``), se construye el espacio de
    búsqueda únicamente para esa familia (modo de estudio por modelo).

    Parámetros
    ----------
    trial : optuna.trial.Trial
        Objeto de ensayo de Optuna.
    model_family : str, opcional
        Si se proporciona, restringe la búsqueda a esa familia.
        Opciones: ``'LSTM'``, ``'GRU'``, ``'CNN-LSTM'``, ``'TCN'``,
        ``'Transformer'``, ``'PatchTST'``, ``'xLSTM'``, ``'RandomForest'``.

    Retorna
    -------
    Dict[str, Any]
        Diccionario con ``model_type`` y todos los hiperparámetros sugeridos
        por Optuna (arquitectura + optimizador).

    Raises
    ------
    ValueError
        Si ``model_family`` no corresponde a ninguna familia conocida.
    """
    _SAMPLERS = {
        "RandomForest": sample_random_forest,
        "LSTM": sample_lstm,
        "GRU": sample_gru,
        "CNN-LSTM": sample_cnn_lstm,
        "TCN": sample_tcn,
        "Transformer": sample_transformer,
        "PatchTST": sample_patchtst,
        "xLSTM": sample_xlstm,
    }

    if model_family is None:
        model_family = trial.suggest_categorical(
            "model_family",
            list(_SAMPLERS.keys()),
        )

    if model_family not in _SAMPLERS:
        raise ValueError(
            f"Familia de modelos '{model_family}' no reconocida. "
            f"Opciones válidas: {list(_SAMPLERS.keys())}."
        )

    return _SAMPLERS[model_family](trial)
