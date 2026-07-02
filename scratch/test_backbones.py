"""Test real de carga y ejecucion de backbones (MOMENT y Chronos)."""
import sys
sys.path.insert(0, 'fatigueset-lib')

import torch
import numpy as np

from fatigueset.models.foundation import (
    MOMENTFatigueRegressor,
    ChronosZeroShotEvaluator,
    MOMENT_LOCAL,
    CHRONOS_LOCAL,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# 1. Probar MOMENT (con version pequeña)
try:
    print("\n--- Cargando MOMENT (small) ---")
    model = MOMENTFatigueRegressor(
        checkpoint=MOMENT_LOCAL,
        n_channels=2,
        seq_len=128,
        output_size=2,
        freeze_backbone=True,
    )
    # Cargar backbone (descarga desde Hugging Face si no existe)
    model.load_backbone()
    model = model.to(device)
    
    # Hacer un forward pass de prueba
    x = torch.randn(2, 128, 2).to(device)
    out = model(x)
    print(f"[MOMENT OK] Output shape: {out.shape} (esperado: [2, 2])")
except Exception as e:
    print(f"[MOMENT FAILED] {e}")

# 2. Probar Chronos (con version tiny)
try:
    print("\n--- Cargando Chronos (tiny) ---")
    evaluator = ChronosZeroShotEvaluator(
        checkpoint=CHRONOS_LOCAL,
        prediction_length=1,
        num_samples=10,
        device=device,
    )
    # Cargar pipeline
    evaluator._load_pipeline()
    
    # Hacer una extraccion de caracteristicas de prueba
    X = np.random.randn(2, 128, 2).astype(np.float32)
    feats = evaluator.extract_features(X, batch_size=2)
    print(f"[Chronos OK] Features shape: {feats.shape} (esperado: [2, 2])")
except Exception as e:
    print(f"[Chronos FAILED] {e}")
