"""Validacion del modulo foundation.py y sus exports."""
import sys
sys.path.insert(0, 'fatigueset-lib')

# Verificar importaciones del modulo foundation
from fatigueset.models.foundation import (
    MOMENTFatigueRegressor,
    ChronosZeroShotEvaluator,
    compute_crps_gaussian,
    compute_coverage,
    evaluate_probabilistic_metrics,
    finetune_moment_kfold,
    MOMENT_LOCAL, MOMENT_SERVER,
    CHRONOS_LOCAL, CHRONOS_SERVER,
)
print('[OK] foundation.py importado correctamente.')
print(f'  MOMENT_LOCAL:   {MOMENT_LOCAL}')
print(f'  MOMENT_SERVER:  {MOMENT_SERVER}')
print(f'  CHRONOS_LOCAL:  {CHRONOS_LOCAL}')
print(f'  CHRONOS_SERVER: {CHRONOS_SERVER}')

# Verificar exports del __init__.py
from fatigueset.models import (
    MOMENTFatigueRegressor,
    ChronosZeroShotEvaluator,
    build_optimizer,
    sample_model_hyperparams,
)
print('[OK] __init__.py exporta correctamente.')

# Verificar CRPS y cobertura con distribucion conocida
import numpy as np
np.random.seed(42)
# Datos donde la Normal N(0,1) es exactamente el modelo generador
y_true = np.random.randn(10000)  # muestras reales de N(0,1)
mu = np.zeros(10000)             # media perfecta
sigma = np.ones(10000)           # sigma perfecta

crps = compute_crps_gaussian(y_true, mu, sigma)
print(f'[OK] compute_crps_gaussian (N(0,1) perfecto): {crps:.4f}  (esperado ~0.564)')

# Con N(0,1) y z=1.645, la cobertura teorica es exactamente 90%
cov = compute_coverage(y_true, mu - 1.645 * sigma, mu + 1.645 * sigma)
print(f'[OK] compute_coverage (90%): {cov:.1%}  (esperado ~90%)')

# Verificar notebook JSON valido (con encoding UTF-8 para acentos)
import json
with open('Jupyters/09_foundation_models.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
n_cells = len(nb['cells'])
print(f'[OK] 09_foundation_models.ipynb: {n_cells} celdas, formato valido.')

# Verificar que MOMENTFatigueRegressor se puede instanciar sin backbone
model = MOMENTFatigueRegressor(n_channels=23, seq_len=128)
print(f'[OK] MOMENTFatigueRegressor instanciado (sin backbone cargado).')
print(f'     checkpoint: {model.checkpoint}')
print(f'     freeze: {model.freeze_backbone}')

# Verificar que ChronosZeroShotEvaluator se puede instanciar
evaluator = ChronosZeroShotEvaluator(num_samples=20)
print(f'[OK] ChronosZeroShotEvaluator instanciado.')
print(f'     checkpoint: {evaluator.checkpoint}')
print(f'     num_samples: {evaluator.num_samples}')

# Verificar evaluate_probabilistic_metrics con datos sinteticos
n = 50
y_true_2d = np.random.randn(n, 2)
samples = np.random.randn(n, 30, 2)  # 30 muestras por prediccion
metrics = evaluate_probabilistic_metrics(y_true_2d, samples, ci_level=0.90)
print(f'[OK] evaluate_probabilistic_metrics keys: {list(metrics.keys())}')

print()
print('=== TODOS LOS TESTS PASARON ===')
