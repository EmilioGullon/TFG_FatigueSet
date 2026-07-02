"""
Script de validación integral del módulo de optimizadores.
Verifica que:
1. Todos los optimizadores se instancian correctamente.
2. Todas las funciones sample_* generan hiperparámetros válidos.
3. build_optimizer + instantiate_model + training step funciona end-to-end.
"""
import sys
sys.path.insert(0, 'fatigueset-lib')

import torch
import torch.nn as nn
import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)

from fatigueset.models import (
    CustomLSTMRegressor,
    CustomGRURegressor,
    CustomCNNLSTMRegressor,
    CustomTCNRegressor,
    CustomTSTransformerRegressor,
    CustomPatchTSTRegressor,
    CustomxLSTMRegressor,
    build_optimizer,
    sample_model_hyperparams,
)
from fatigueset.models.optimizers import (
    sample_lstm, sample_gru, sample_cnn_lstm,
    sample_tcn, sample_transformer, sample_patchtst, sample_xlstm,
)

N_FEATURES = 23
SEQ_LEN = 32
BATCH_SIZE = 4


def instantiate_model(hyperparams, n_features):
    """Factory de modelos para el test."""
    mt = hyperparams["model_type"]
    if mt == "LSTM":
        return CustomLSTMRegressor(
            input_size=n_features,
            hidden_size=hyperparams["hidden_size"],
            num_layers=hyperparams["num_layers"],
            dropout=hyperparams["dropout"],
        )
    elif mt == "GRU":
        return CustomGRURegressor(
            input_size=n_features,
            hidden_size=hyperparams["hidden_size"],
            num_layers=hyperparams["num_layers"],
            dropout=hyperparams["dropout"],
        )
    elif mt == "CNN-LSTM":
        return CustomCNNLSTMRegressor(
            input_size=n_features,
            conv_channels=hyperparams["conv_channels"],
            kernel_size=hyperparams["kernel_size"],
            pool_size=hyperparams["pool_size"],
            hidden_size=hyperparams["hidden_size"],
            num_layers=hyperparams["num_layers"],
            dropout=hyperparams["dropout"],
        )
    elif mt == "TCN":
        return CustomTCNRegressor(
            input_size=n_features,
            num_channels=hyperparams["num_channels"],
            kernel_size=hyperparams["kernel_size"],
            dropout=hyperparams["dropout"],
        )
    elif mt == "Transformer":
        return CustomTSTransformerRegressor(
            input_size=n_features,
            d_model=hyperparams["d_model"],
            num_heads=hyperparams["num_heads"],
            num_layers=hyperparams["num_layers"],
            dim_feedforward=hyperparams["dim_feedforward"],
            dropout=hyperparams["dropout"],
        )
    elif mt == "PatchTST":
        return CustomPatchTSTRegressor(
            input_size=n_features,
            patch_len=hyperparams["patch_len"],
            stride=hyperparams["stride"],
            d_model=hyperparams["d_model"],
            num_heads=hyperparams["num_heads"],
            num_layers=hyperparams["num_layers"],
            dim_feedforward=hyperparams["dim_feedforward"],
            dropout=hyperparams["dropout"],
        )
    elif mt == "xLSTM":
        return CustomxLSTMRegressor(
            input_size=n_features,
            hidden_size=hyperparams["hidden_size"],
            num_layers=hyperparams["num_layers"],
            dropout=hyperparams["dropout"],
        )
    else:
        raise ValueError(f"Modelo desconocido: {mt}")


SAMPLERS = {
    "LSTM": sample_lstm,
    "GRU": sample_gru,
    "CNN-LSTM": sample_cnn_lstm,
    "TCN": sample_tcn,
    "Transformer": sample_transformer,
    "PatchTST": sample_patchtst,
    "xLSTM": sample_xlstm,
}

print("=" * 60)
print("TEST DE INTEGRACIÓN: optimizers.py")
print("=" * 60)

# Datos de prueba sintéticos
X_dummy = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES)
y_dummy = torch.randn(BATCH_SIZE, 2)

all_ok = True

for model_name, sampler_fn in SAMPLERS.items():
    # Crear estudio Optuna real
    study = optuna.create_study(direction="minimize")

    errors = []

    def objective(trial, sampler_fn=sampler_fn, model_name=model_name):
        hp = sampler_fn(trial)

        # Verificar que el diccionario tiene las claves requeridas
        required_keys = {"model_type", "optimizer", "lr", "weight_decay"}
        missing = required_keys - set(hp.keys())
        if missing:
            raise ValueError(f"{model_name}: faltan claves {missing}")

        # Instanciar modelo
        model = instantiate_model(hp, N_FEATURES)
        assert model is not None

        # Construir optimizador
        opt = build_optimizer(
            model=model,
            optimizer_name=hp["optimizer"],
            lr=hp["lr"],
            weight_decay=hp["weight_decay"],
            momentum=hp.get("momentum", 0.9),
            alpha=hp.get("alpha", 0.99),
        )

        # Un paso de entrenamiento completo
        model.train()
        loss_fn = nn.MSELoss()
        opt.zero_grad()
        pred = model(X_dummy)
        loss = loss_fn(pred, y_dummy)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        return loss.item()

    try:
        study.optimize(objective, n_trials=2)
        print(f"  [OK] {model_name}")
        print(f"       Mejor MSE: {study.best_value:.4f}")

    except Exception as e:
        print(f"  [ERROR] {model_name}: {e}")
        all_ok = False

# Test de la función unificada sample_model_hyperparams
print("\n--- Test sample_model_hyperparams (función unificada) ---")
for family in ["LSTM", "GRU", "CNN-LSTM", "TCN", "Transformer", "PatchTST", "xLSTM"]:
    study = optuna.create_study()
    def obj_unified(trial, f=family):
        hp = sample_model_hyperparams(trial, model_family=f)
        assert hp["model_type"] is not None
        return 0.0
    study.optimize(obj_unified, n_trials=1)
    print(f"  [OK] sample_model_hyperparams(family={family})")

# Test de error esperado
try:
    study = optuna.create_study()
    study.optimize(lambda t: sample_model_hyperparams(t, "ModeloInexistente"), n_trials=1)
    print("  [ERROR] No se lanzó ValueError para familia desconocida")
    all_ok = False
except Exception:
    print("  [OK] ValueError correctamente lanzado para familia desconocida")

print("\n" + "=" * 60)
if all_ok:
    print("RESULTADO: TODOS LOS TESTS PASARON CORRECTAMENTE")
else:
    print("RESULTADO: ALGUNOS TESTS FALLARON - VER DETALLES ARRIBA")
print("=" * 60)
