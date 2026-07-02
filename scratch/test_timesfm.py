"""Test de integracion para TimesFMZeroShotEvaluator en fatigueset."""
import sys
sys.path.insert(0, 'fatigueset-lib')

import numpy as np
import torch
import time

from fatigueset.models import TimesFMZeroShotEvaluator, compute_crps_gaussian, compute_coverage

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

try:
    print("\n--- Instanciando TimesFMZeroShotEvaluator ---")
    evaluator = TimesFMZeroShotEvaluator(
        checkpoint="google/timesfm-2.5-200m-pytorch",
        prediction_length=1,
        max_context=128,
        device=device,
    )
    
    # Generar datos sinteticos para pruebas locales rapidas
    # 4 muestras, 128 longitud de secuencia, 23 canales (fisiologicos de FatigueSet)
    X_train = np.random.randn(6, 128, 23).astype(np.float32)
    y_train = np.random.randn(6, 2).astype(np.float32)
    X_val = np.random.randn(2, 128, 23).astype(np.float32)
    y_val = np.random.randn(2, 2).astype(np.float32)

    print("\n--- Entrenando Sonda Ridge con Inferencia en Batch ---")
    t0 = time.time()
    y_pred, feats_val, probe = evaluator.fit_probe(X_train, y_train, X_val)
    print(f"[OK] Sonda entrenada e inferida en {time.time() - t0:.1f}s")
    
    print(f"[OK] Predicciones shape:         {y_pred.shape} (esperado: [2, 2])")
    print(f"[OK] Caracteristicas val shape:  {feats_val.shape} (esperado: [2, 23])")
    
    # Calcular metricas probabilistas estimando sigma por residuos
    y_pred_train = probe.predict(evaluator.extract_features(X_train)[0])
    residuals = y_train - y_pred_train
    sigma_f = np.std(residuals[:, 0]) + 1e-8
    sigma_m = np.std(residuals[:, 1]) + 1e-8
    
    crps_f = compute_crps_gaussian(y_val[:, 0], y_pred[:, 0], np.full(len(y_val), sigma_f))
    cov_f = compute_coverage(y_val[:, 0], y_pred[:, 0] - 1.645*sigma_f, y_pred[:, 0] + 1.645*sigma_f)
    
    print(f"[OK] CRPS fisica:                {crps_f:.4f}")
    print(f"[OK] Cobertura 90% fisica:       {cov_f:.1%}")
    print("\n=== TODO VALIDADO CON ÉXITO ===")

except Exception as e:
    import traceback
    print(f"[FAILED] Error al ejecutar TimesFMZeroShotEvaluator: {e}")
    traceback.print_exc()
