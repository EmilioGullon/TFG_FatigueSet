"""Comparar modelos usando resultados guardados por run_models_classicos.py
Genera `output/compare_models.png` y `output/compare_bootstrap.json`.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def bootstrap_diff(a, b, n_iter=1000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = []
    n = min(len(a), len(b))
    for _ in range(n_iter):
        ia = rng.choice(a, size=n, replace=True)
        ib = rng.choice(b, size=n, replace=True)
        diffs.append(np.mean(ia) - np.mean(ib))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


def main(results_dir='output'):
    out = Path(results_dir)
    cv_pkl = out / 'cv_scores.pkl'
    if not cv_pkl.exists():
        raise FileNotFoundError(f"No existe {cv_pkl}. Ejecuta run_models_classicos.py primero.")

    import pickle
    with open(cv_pkl, 'rb') as f:
        cv_scores = pickle.load(f)

    # Ordenar por media
    means = {k: np.mean(v) for k, v in cv_scores.items()}
    ordered = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
    if len(ordered) < 2:
        raise RuntimeError("Se necesitan al menos 2 modelos para comparar.")
    top2 = [ordered[0][0], ordered[1][0]]

    a = np.array(cv_scores[top2[0]])
    b = np.array(cv_scores[top2[1]])

    mean_diff, lo, hi = bootstrap_diff(a, b, n_iter=2000, seed=42)

    summary = {
        'top1': top2[0],
        'top2': top2[1],
        'mean_diff': mean_diff,
        'ci95_lo': lo,
        'ci95_hi': hi
    }

    (out / 'compare_bootstrap.json').write_text(json.dumps(summary, indent=2))

    # Figura: distribuciones de CV R2
    plt.figure(figsize=(8, 4))
    plt.boxplot([cv_scores[top2[0]], cv_scores[top2[1]]], labels=top2)
    plt.title('Distribución CV R² - Top2 modelos')
    plt.ylabel('R²')
    plt.tight_layout()
    plt.savefig(out / 'compare_models.png')
    print(f"✓ Comparación generada en {out}")


if __name__ == '__main__':
    main()
