"""
experimento_optuna_v2.py
========================
Experimento Optuna v2 con persistencia SQLite.

Características:
- Dos targets por separado: Fatiga Física (col 0) y Fatiga Mental (col 1)
- SQLite storage para reanudar entre ejecuciones: optuna_v2.db
- 3-fold GroupKFold por sujeto (consistente con el resto del TFG)
- SERVER_MODE: N_TRIALS=50, EPOCHS_PER_TRIAL=20, FINAL_EPOCHS=80
- Generación de 8 tipos de gráficas de análisis
"""

import matplotlib
matplotlib.use('Agg')

# ── Modo de ejecución ────────────────────────────────────────────────────────
SERVER_MODE = False  # sed cambia esto a True en el script SLURM

# ── Importaciones ────────────────────────────────────────────────────────────
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os, gc, json, time, warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.importance import get_param_importances

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Path al paquete ──────────────────────────────────────────────────────────
lib_path = str(Path(__file__).resolve().parent.parent / "fatigueset-lib")
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

from fatigueset import FatigueSetPipeline
from fatigueset.models import (
    FatigueSequenceDataset,
    CustomLSTMRegressor,
    CustomGRURegressor,
    CustomCNNLSTMRegressor,
    CustomTCNRegressor,
    CustomTSTransformerRegressor,
    CustomPatchTSTRegressor,
    CustomxLSTMRegressor,
)
from fatigueset.models.rnn import (
    FatigueSequenceDataset as FSD,
    _prepare_target_table,
    _merge_raw_streams,
    _build_sequences,
)
from fatigueset.models.optimizers import (
    build_optimizer,
    sample_lstm, sample_gru, sample_cnn_lstm, sample_tcn,
    sample_transformer, sample_patchtst, sample_xlstm,
)

# ── Semillas y dispositivo ───────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Dispositivo: {device}")
print(f"[INFO] SERVER_MODE: {SERVER_MODE}")

# ── Configuración del experimento ────────────────────────────────────────────
if SERVER_MODE:
    N_TRIALS         = 50     # trials por modelo x target
    N_CV_SPLITS      = 3      # GroupKFold folds en cada trial
    EPOCHS_PER_TRIAL = 20     # épocas rápidas por trial
    FINAL_EPOCHS     = 80     # épocas en evaluación final
    PATIENCE         = 15     # Early Stopping en evaluación final
    TIMEOUT_SECONDS  = int(10.5 * 3600)  # 10.5h para dejar margen a gráficas
else:
    N_TRIALS         = 2
    N_CV_SPLITS      = 2
    EPOCHS_PER_TRIAL = 2
    FINAL_EPOCHS     = 5
    PATIENCE         = 3
    TIMEOUT_SECONDS  = None

SEQ_LEN = 128
STEP    = 64

TARGETS = {
    "Fatiga_Fisica":  0,  # columna 0 de y_arr
    "Fatiga_Mental":  1,  # columna 1 de y_arr
}

MODEL_SAMPLERS = {
    "Custom LSTM":        sample_lstm,
    "Custom GRU":         sample_gru,
    "Custom CNN-LSTM":    sample_cnn_lstm,
    "Custom TCN":         sample_tcn,
    "Custom Transformer": sample_transformer,
    "Custom PatchTST":    sample_patchtst,
    "Custom xLSTM":       sample_xlstm,
}

PALETTE = {
    "Custom LSTM":        "#4c72b0",
    "Custom GRU":         "#dd8452",
    "Custom CNN-LSTM":    "#55a868",
    "Custom TCN":         "#c44e52",
    "Custom Transformer": "#8172b2",
    "Custom PatchTST":    "#937860",
    "Custom xLSTM":       "#da8bc3",
}
MODEL_NAMES = list(MODEL_SAMPLERS.keys())

# ── Directorios de salida ─────────────────────────────────────────────────────
script_dir  = Path(__file__).resolve().parent
root_dir    = script_dir.parent
output_dir  = root_dir / "output" / "optuna_v2"
output_dir.mkdir(parents=True, exist_ok=True)
db_path     = script_dir / "optuna_v2.db"
storage_url = f"sqlite:///{db_path}"

print(f"[INFO] Output dir: {output_dir}")
print(f"[INFO] SQLite DB : {db_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

dataset_path = str(root_dir / "fatigueset")
print("\n[1/4] Cargando dataset FatigueSet...")
pipeline   = FatigueSetPipeline(dataset_path=dataset_path, umbral_nulos=5.0)
raw        = pipeline.cargar_dataset(verbose=False)
df_ml      = pipeline.construir_dataset_ml(raw)
df_targets = _prepare_target_table(df_ml)

print("[1/4] Fusionando streams temporales (chest + wrist)...")
df_raw = _merge_raw_streams(raw)

print("[1/4] Construyendo secuencias...")
X_arr, y_arr, groups, feature_cols = _build_sequences(
    df_raw=df_raw,
    df_targets=df_targets,
    seq_len=SEQ_LEN,
    step=STEP,
)

N_FEATURES = len(feature_cols)
N_WINDOWS  = len(X_arr)
print(f"[OK] X: {X_arr.shape} | y: {y_arr.shape} | grupos: {len(np.unique(groups))}")


# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════════

def _cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def instantiate_model(hyperparams: Dict[str, Any], n_features: int, output_size: int = 1) -> nn.Module:
    """Instancia el modelo DL con output_size=1 para un único target."""
    mt  = hyperparams["model_type"]
    inp = n_features
    if mt == "LSTM":
        m = CustomLSTMRegressor(
            input_size=inp, hidden_size=hyperparams["hidden_size"],
            num_layers=hyperparams["num_layers"], dropout=hyperparams["dropout"])
    elif mt == "GRU":
        m = CustomGRURegressor(
            input_size=inp, hidden_size=hyperparams["hidden_size"],
            num_layers=hyperparams["num_layers"], dropout=hyperparams["dropout"])
    elif mt == "CNN-LSTM":
        m = CustomCNNLSTMRegressor(
            input_size=inp, conv_channels=hyperparams["conv_channels"],
            kernel_size=hyperparams["kernel_size"], pool_size=hyperparams["pool_size"],
            hidden_size=hyperparams["hidden_size"], num_layers=hyperparams["num_layers"],
            dropout=hyperparams["dropout"])
    elif mt == "TCN":
        m = CustomTCNRegressor(
            input_size=inp, num_channels=hyperparams["num_channels"],
            kernel_size=hyperparams["kernel_size"], dropout=hyperparams["dropout"])
    elif mt == "Transformer":
        m = CustomTSTransformerRegressor(
            input_size=inp, d_model=hyperparams["d_model"],
            num_heads=hyperparams["num_heads"], num_layers=hyperparams["num_layers"],
            dim_feedforward=hyperparams["dim_feedforward"], dropout=hyperparams["dropout"])
    elif mt == "PatchTST":
        m = CustomPatchTSTRegressor(
            input_size=inp, patch_len=hyperparams["patch_len"],
            stride=hyperparams["stride"], d_model=hyperparams["d_model"],
            num_heads=hyperparams["num_heads"], num_layers=hyperparams["num_layers"],
            dim_feedforward=hyperparams["dim_feedforward"], dropout=hyperparams["dropout"])
    elif mt == "xLSTM":
        m = CustomxLSTMRegressor(
            input_size=inp, hidden_size=hyperparams["hidden_size"],
            num_layers=hyperparams["num_layers"], dropout=hyperparams["dropout"])
    else:
        raise ValueError(f"Tipo de modelo desconocido: {mt}")
    # Reemplazar capa de salida a 1 dimensión si es necesario
    if hasattr(m, 'fc') and m.fc.out_features != output_size:
        m.fc = nn.Linear(m.fc.in_features, output_size)
    return m


def quick_cv_eval(
    hyperparams: Dict[str, Any],
    target_col: int,
    n_splits: int = 3,
    epochs: int = 10,
    trial=None,
) -> float:
    """Evaluación rápida GroupKFold para un trial de Optuna. Devuelve MSE medio de validación."""
    batch_size = hyperparams.get("batch_size", 32)
    kf         = GroupKFold(n_splits=n_splits)
    val_losses = []
    loss_fn    = nn.MSELoss()

    for fold_idx, (train_idx, val_idx) in enumerate(
        kf.split(np.arange(N_WINDOWS), groups=groups)
    ):
        X_tr = X_arr[train_idx]
        y_tr = y_arr[train_idx, target_col:target_col + 1]
        X_va = X_arr[val_idx]
        y_va = y_arr[val_idx, target_col:target_col + 1]

        ds_tr = FSD(X_tr, y_tr)
        ds_va = FSD(X_va, y_va)
        tr_ld = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
        va_ld = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        model = instantiate_model(hyperparams, N_FEATURES, output_size=1).to(device)
        opt   = build_optimizer(
            model, hyperparams.get("optimizer", "Adam"),
            lr=hyperparams.get("lr", 1e-3),
            weight_decay=hyperparams.get("weight_decay", 0.0),
            momentum=hyperparams.get("momentum", 0.9),
            alpha=hyperparams.get("alpha", 0.99),
        )

        for _ in range(epochs):
            model.train()
            for xb, yb in tr_ld:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

        model.eval()
        total = 0.0
        with torch.no_grad():
            for xb, yb in va_ld:
                xb, yb = xb.to(device), yb.to(device)
                total += loss_fn(model(xb), yb).item()
        fold_mse = total / max(len(va_ld), 1)
        val_losses.append(fold_mse)

        del model, opt, tr_ld, va_ld, ds_tr, ds_va
        _cleanup_gpu()

        if trial is not None:
            trial.report(float(np.mean(val_losses)), step=fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return float(np.mean(val_losses))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BÚSQUEDA OPTUNA CON SQLITE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[2/4] Búsqueda bayesiana de hiperparámetros con Optuna (SQLite)...")

best_hyperparams:     Dict[str, Dict[str, Any]] = {}
optuna_study_results: Dict[str, Dict[str, Any]] = {}
all_study_data:       Dict[str, Dict[str, Any]] = {}  # {target: {model: study}}

for target_name, target_col in TARGETS.items():
    print(f"\n{'='*65}")
    print(f" TARGET: {target_name.replace('_', ' ')}")
    print(f"{'='*65}")

    best_hyperparams[target_name]     = {}
    optuna_study_results[target_name] = {}
    all_study_data[target_name]       = {}

    for model_name, sampler_fn in MODEL_SAMPLERS.items():
        study_name = f"{target_name}__{model_name.replace(' ', '_')}"
        print(f"\n  ► {model_name} [{target_name}]")

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction="minimize",
            sampler=TPESampler(seed=SEED),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
            load_if_exists=True,
        )

        already_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining    = max(0, N_TRIALS - already_done)
        print(f"     Trials previos: {already_done} | Por ejecutar: {remaining}")

        def make_objective(sfn, tcol):
            def objective(trial):
                hp = sfn(trial)
                return quick_cv_eval(hp, target_col=tcol,
                                     n_splits=N_CV_SPLITS,
                                     epochs=EPOCHS_PER_TRIAL,
                                     trial=trial)
            return objective

        t0 = time.time()
        study.optimize(
            make_objective(sampler_fn, target_col),
            n_trials=remaining,
            timeout=TIMEOUT_SECONDS,
            show_progress_bar=False,
            gc_after_trial=True,
        )
        elapsed = time.time() - t0

        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_pruned   = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        print(f"     Total: {n_complete} completados, {n_pruned} podados | {elapsed:.0f}s")
        print(f"     Mejor MSE validación: {study.best_value:.4f}")

        best_hp = sampler_fn(study.best_trial)
        best_hyperparams[target_name][model_name]     = best_hp
        all_study_data[target_name][model_name]       = study
        optuna_study_results[target_name][model_name] = {
            "Modelo":             model_name,
            "Target":             target_name,
            "Mejor MSE (Optuna)": study.best_value,
            "Trials completados": n_complete,
            "Trials podados":     n_pruned,
            "Optimizador":        best_hp.get("optimizer", "N/A"),
            "LR":                 best_hp.get("lr", float("nan")),
            "Tiempo (s)":         elapsed,
        }

rows_optuna = [v for td in optuna_study_results.values() for v in td.values()]
pd.DataFrame(rows_optuna).to_csv(output_dir / "resultados_busqueda_optuna_v2.csv", index=False)
print(f"\n[OK] Tabla de búsqueda guardada.")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EVALUACIÓN FINAL CON MEJORES HIPERPARÁMETROS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[3/4] Evaluación final ({N_CV_SPLITS}-fold GroupKFold, {FINAL_EPOCHS} epochs + ES)...")

all_learning_curves: Dict[str, Dict[str, Dict[str, list]]] = {}
all_fold_metrics:    Dict[str, Dict[str, Dict[str, list]]] = {}
all_preds_targets:   Dict[str, Dict[str, Tuple]]           = {}
final_results: List[Dict] = []
kf_final = GroupKFold(n_splits=N_CV_SPLITS)
loss_fn  = nn.MSELoss()

for target_name, target_col in TARGETS.items():
    all_learning_curves[target_name] = {}
    all_fold_metrics[target_name]    = {}
    all_preds_targets[target_name]   = {}
    print(f"\n  TARGET: {target_name}")

    for model_name, best_hp in best_hyperparams[target_name].items():
        print(f"   ► {model_name}...", end=" ", flush=True)

        fold_train_losses, fold_val_losses = [], []
        fold_mae, fold_rmse, fold_r2       = [], [], []
        all_y_true, all_y_pred             = [], []

        for fold_idx, (tr_idx, va_idx) in enumerate(
            kf_final.split(np.arange(N_WINDOWS), groups=groups)
        ):
            X_tr = X_arr[tr_idx]
            y_tr = y_arr[tr_idx, target_col:target_col + 1]
            X_va = X_arr[va_idx]
            y_va = y_arr[va_idx, target_col:target_col + 1]

            ds_tr = FSD(X_tr, y_tr)
            ds_va = FSD(X_va, y_va)
            bs    = best_hp.get("batch_size", 32)
            tr_ld = DataLoader(ds_tr, batch_size=bs, shuffle=True,  num_workers=0, pin_memory=True)
            va_ld = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)

            model = instantiate_model(best_hp, N_FEATURES, output_size=1).to(device)
            opt   = build_optimizer(
                model, best_hp.get("optimizer", "Adam"),
                lr=best_hp.get("lr", 1e-3),
                weight_decay=best_hp.get("weight_decay", 0.0),
                momentum=best_hp.get("momentum", 0.9),
                alpha=best_hp.get("alpha", 0.99),
            )

            train_curve, val_curve = [], []
            best_val, patience_cnt = float("inf"), 0

            for epoch in range(FINAL_EPOCHS):
                model.train()
                ep_loss = 0.0
                for xb, yb in tr_ld:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    out  = model(xb)
                    loss = loss_fn(out, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    ep_loss += loss.item()
                train_curve.append(ep_loss / max(len(tr_ld), 1))

                model.eval()
                v_loss = 0.0
                with torch.no_grad():
                    for xb, yb in va_ld:
                        xb, yb = xb.to(device), yb.to(device)
                        v_loss += loss_fn(model(xb), yb).item()
                v_loss /= max(len(va_ld), 1)
                val_curve.append(v_loss)

                if v_loss < best_val:
                    best_val, patience_cnt = v_loss, 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= PATIENCE:
                        break

            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for xb, yb in va_ld:
                    preds.append(model(xb.to(device)).cpu().numpy())
                    trues.append(yb.numpy())

            p_flat = np.vstack(preds).flatten()
            t_flat = np.vstack(trues).flatten()
            all_y_true.extend(t_flat.tolist())
            all_y_pred.extend(p_flat.tolist())

            fold_mae.append(mean_absolute_error(t_flat, p_flat))
            fold_rmse.append(np.sqrt(mean_squared_error(t_flat, p_flat)))
            fold_r2.append(r2_score(t_flat, p_flat))
            fold_train_losses.append(train_curve)
            fold_val_losses.append(val_curve)

            del model, opt, tr_ld, va_ld, ds_tr, ds_va
            _cleanup_gpu()

        all_learning_curves[target_name][model_name] = {
            "train": fold_train_losses,
            "val":   fold_val_losses,
        }
        all_fold_metrics[target_name][model_name] = {
            "mae":  fold_mae,
            "rmse": fold_rmse,
            "r2":   fold_r2,
        }
        all_preds_targets[target_name][model_name] = (
            np.array(all_y_true), np.array(all_y_pred)
        )

        num_params = sum(
            p.numel() for p in instantiate_model(best_hp, N_FEATURES).parameters()
            if p.requires_grad
        )
        final_results.append({
            "Modelo":        model_name,
            "Target":        target_name,
            "MAE Medio":     float(np.mean(fold_mae)),
            "MAE Std":       float(np.std(fold_mae)),
            "RMSE Medio":    float(np.mean(fold_rmse)),
            "RMSE Std":      float(np.std(fold_rmse)),
            "R² Medio":      float(np.mean(fold_r2)),
            "R² Std":        float(np.std(fold_r2)),
            "Optimizador":   best_hp.get("optimizer", "N/A"),
            "LR":            best_hp.get("lr", float("nan")),
            "Nº Parámetros": num_params,
        })
        print(f"MAE={np.mean(fold_mae):.3f} | RMSE={np.mean(fold_rmse):.3f} | R²={np.mean(fold_r2):.3f}")

df_final = pd.DataFrame(final_results)
df_final.to_csv(output_dir / "comparativa_optuna_v2.csv", index=False)

def _to_native(obj):
    if isinstance(obj, dict):         return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):         return [_to_native(v) for v in obj]
    if isinstance(obj, np.integer):   return int(obj)
    if isinstance(obj, np.floating):  return float(obj)
    return obj

with open(output_dir / "mejores_hiperparametros_v2.json", "w", encoding="utf-8") as f:
    json.dump(_to_native(best_hyperparams), f, indent=2, ensure_ascii=False)

print(f"\n[OK] Resultados finales guardados.")
print(df_final.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GENERACIÓN DE GRÁFICAS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[4/4] Generando gráficas...")


def _save(fig, name):
    path = output_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   [OK] {name}")


# ── Gráfica 1: Comparativa general — MAE, RMSE, R² × target ─────────────────
def plot_comparativa_general():
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(
        "Comparativa General de Modelos — FatigueSet\n"
        "(Evaluación 3-fold GroupKFold, hiperparámetros optimizados con Optuna)",
        fontsize=13, fontweight="bold",
    )
    for row, (target_name, _) in enumerate(TARGETS.items()):
        target_label = target_name.replace("_", " ")
        df_t  = df_final[df_final["Target"] == target_name].copy()
        names = df_t["Modelo"].tolist()
        cols  = [PALETTE.get(m, "#999") for m in names]
        x     = np.arange(len(names))

        for col, (mk, sk, title) in enumerate([
            ("MAE Medio",  "MAE Std",  "MAE"),
            ("RMSE Medio", "RMSE Std", "RMSE"),
            ("R² Medio",   "R² Std",   "R²"),
        ]):
            ax   = axes[row, col]
            vals = df_t[mk].values
            stds = df_t[sk].values
            ax.bar(x, vals, color=cols, alpha=0.85, edgecolor="white", linewidth=0.5)
            ax.errorbar(x, vals, yerr=stds, fmt="none", color="black", capsize=4, linewidth=1.2)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [m.replace("Custom ", "") for m in names], rotation=30, ha="right", fontsize=9
            )
            ax.set_ylabel(title, fontsize=10)
            ax.set_title(f"{title} — {target_label}", fontsize=10, fontweight="bold")
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if mk == "R² Medio":
                ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    plt.tight_layout()
    _save(fig, "01_comparativa_general.png")


# ── Gráfica 2: Learning curves por modelo ────────────────────────────────────
def plot_learning_curves():
    from matplotlib.lines import Line2D
    n_m = len(MODEL_NAMES)
    n_t = len(TARGETS)
    fig, axes = plt.subplots(n_t, n_m, figsize=(5 * n_m, 4 * n_t), squeeze=False)
    fig.suptitle("Learning Curves — MSE Loss por Modelo y Target", fontsize=13, fontweight="bold")

    for row, (target_name, _) in enumerate(TARGETS.items()):
        tl = target_name.replace("_", " ")
        for col, model_name in enumerate(MODEL_NAMES):
            ax     = axes[row, col]
            curves = all_learning_curves[target_name].get(model_name, {})
            color  = PALETTE.get(model_name, "#555")
            for fi, (tr, va) in enumerate(
                zip(curves.get("train", []), curves.get("val", []))
            ):
                a  = 0.35 if fi < len(curves.get("train", [])) - 1 else 1.0
                lw = 1.0  if fi < len(curves.get("train", [])) - 1 else 2.0
                ax.plot(tr, color=color,   alpha=a, linewidth=lw)
                ax.plot(va, color="black", alpha=a, linewidth=lw, linestyle="--")
            ax.set_title(f"{model_name.replace('Custom ', '')} | {tl}", fontsize=8, fontweight="bold")
            ax.set_xlabel("Época", fontsize=8)
            ax.set_ylabel("MSE", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.legend(
        handles=[
            Line2D([0], [0], color="gray",  linestyle="-",  label="Train"),
            Line2D([0], [0], color="black", linestyle="--", label="Validación"),
        ],
        loc="lower center", ncol=2, fontsize=10, frameon=False,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _save(fig, "02_learning_curves.png")


# ── Gráfica 3: Convergencia Optuna ───────────────────────────────────────────
def plot_optuna_convergencia():
    n_m = len(MODEL_NAMES)
    n_t = len(TARGETS)
    fig, axes = plt.subplots(n_t, n_m, figsize=(5 * n_m, 4 * n_t), squeeze=False)
    fig.suptitle("Convergencia de Optuna — MSE de Validación vs Nº Trial",
                 fontsize=13, fontweight="bold")

    for row, (target_name, _) in enumerate(TARGETS.items()):
        tl = target_name.replace("_", " ")
        for col, model_name in enumerate(MODEL_NAMES):
            ax    = axes[row, col]
            study = all_study_data[target_name].get(model_name)
            color = PALETTE.get(model_name, "#555")
            if study is None:
                ax.set_visible(False)
                continue
            trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not trials:
                ax.set_visible(False)
                continue
            values = [t.value for t in trials]
            best_so_far = np.minimum.accumulate(values)
            ax.scatter(range(len(values)), values, color=color, alpha=0.4, s=15, zorder=2)
            ax.plot(range(len(best_so_far)), best_so_far, color=color, linewidth=2.0, zorder=3)
            ax.set_title(f"{model_name.replace('Custom ', '')} | {tl}", fontsize=8, fontweight="bold")
            ax.set_xlabel("Trial nº", fontsize=8)
            ax.set_ylabel("MSE", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save(fig, "03_optuna_convergencia.png")


# ── Gráfica 4: Histogramas de distribución MSE de trials ─────────────────────
def plot_optuna_histogramas():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Distribución de MSE de Trials Optuna por Familia de Modelo",
                 fontsize=13, fontweight="bold")
    for ax, (target_name, _) in zip(axes, TARGETS.items()):
        tl = target_name.replace("_", " ")
        for model_name in MODEL_NAMES:
            study = all_study_data[target_name].get(model_name)
            if study is None:
                continue
            vals = [t.value for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
            if not vals:
                continue
            ax.hist(vals, bins=12, alpha=0.5, color=PALETTE.get(model_name, "#aaa"),
                    label=model_name.replace("Custom ", ""), edgecolor="white")
        ax.set_xlabel("MSE Validación", fontsize=10)
        ax.set_ylabel("Frecuencia", fontsize=10)
        ax.set_title(tl, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, ncol=2, frameon=False)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save(fig, "04_optuna_histogramas.png")


# ── Gráfica 5: Importancia de hiperparámetros ────────────────────────────────
def plot_importancia_hp():
    n_m = len(MODEL_NAMES)
    fig, axes = plt.subplots(2, n_m, figsize=(5 * n_m, 8), squeeze=False)
    fig.suptitle("Importancia de Hiperparámetros — Optuna Feature Importance",
                 fontsize=13, fontweight="bold")

    for row, (target_name, _) in enumerate(TARGETS.items()):
        tl = target_name.replace("_", " ")
        for col, model_name in enumerate(MODEL_NAMES):
            ax    = axes[row, col]
            study = all_study_data[target_name].get(model_name)
            color = PALETTE.get(model_name, "#aaa")
            ax.set_title(f"{model_name.replace('Custom ', '')} | {tl}", fontsize=8, fontweight="bold")
            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if len(completed) < 3:
                ax.text(0.5, 0.5, "Insuficientes\ntrials", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
                continue
            try:
                importance = get_param_importances(study)
                items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:6]
                if items:
                    params, vals = zip(*items)
                    params_clean = [p.split("_", 1)[-1] if "_" in p else p for p in params]
                    ax.barh(list(params_clean)[::-1], list(vals)[::-1], color=color, alpha=0.8)
                ax.set_xlabel("Importancia", fontsize=8)
                ax.tick_params(labelsize=7)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
            ax.grid(axis="x", alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save(fig, "05_importancia_hp.png")


# ── Gráfica 6: Radar chart multidimensional ──────────────────────────────────
def plot_radar_chart():
    categories = ["MAE↓", "RMSE↓", "R²↑", "Velocidad↑", "Params↓"]
    N_cat      = len(categories)
    angles     = np.linspace(0, 2 * np.pi, N_cat, endpoint=False).tolist()
    angles    += angles[:1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw=dict(polar=True))
    fig.suptitle("Radar Chart — Comparativa Multidimensional de Modelos",
                 fontsize=13, fontweight="bold")

    def norm_inv(vals):
        mn, mx = np.min(vals), np.max(vals)
        if mx == mn:
            return np.ones_like(vals, dtype=float)
        return 1.0 - (vals - mn) / (mx - mn)

    def norm_dir(vals):
        mn, mx = np.min(vals), np.max(vals)
        if mx == mn:
            return np.ones_like(vals, dtype=float)
        return (vals - mn) / (mx - mn)

    for ax, (target_name, _) in zip(axes, TARGETS.items()):
        tl   = target_name.replace("_", " ")
        df_t = df_final[df_final["Target"] == target_name].copy().reset_index(drop=True)
        optuna_times = [
            optuna_study_results[target_name].get(mn, {}).get("Tiempo (s)", 1.0) or 1.0
            for mn in df_t["Modelo"]
        ]
        df_t["Tiempo (s)"] = optuna_times

        maes   = norm_inv(df_t["MAE Medio"].values)
        rmses  = norm_inv(df_t["RMSE Medio"].values)
        r2s    = norm_dir(np.clip(df_t["R² Medio"].values, -1, 1))
        speeds = norm_inv(np.array(optuna_times, dtype=float))
        params = norm_inv(df_t["Nº Parámetros"].values.astype(float))

        for i, model_name in enumerate(df_t["Modelo"]):
            values = [maes[i], rmses[i], r2s[i], speeds[i], params[i]] + [maes[i]]
            color  = PALETTE.get(model_name, "#999")
            ax.plot(angles, values, color=color, linewidth=1.8,
                    label=model_name.replace("Custom ", ""))
            ax.fill(angles, values, color=color, alpha=0.06)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7, color="gray")
        ax.set_title(tl, fontsize=11, fontweight="bold", pad=12)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8, frameon=False)

    plt.tight_layout()
    _save(fig, "06_radar_chart.png")


# ── Gráfica 7: Box plot de errores por fold ───────────────────────────────────
def plot_boxplot_cv():
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Box Plot de Errores por Fold de Validación Cruzada",
                 fontsize=13, fontweight="bold")

    for row, (target_name, _) in enumerate(TARGETS.items()):
        tl = target_name.replace("_", " ")
        for col, (mkey, mtitle) in enumerate([("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R²")]):
            ax     = axes[row, col]
            data   = []
            labels = []
            colors = []
            for mn in MODEL_NAMES:
                fm   = all_fold_metrics[target_name].get(mn, {})
                vals = fm.get(mkey, [])
                if vals:
                    data.append(vals)
                    labels.append(mn.replace("Custom ", ""))
                    colors.append(PALETTE.get(mn, "#aaa"))
            if data:
                bp = ax.boxplot(data, patch_artist=True,
                                medianprops=dict(color="black", linewidth=2))
                for patch, c in zip(bp["boxes"], colors):
                    patch.set_facecolor(c)
                    patch.set_alpha(0.75)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel(mtitle, fontsize=10)
            ax.set_title(f"{mtitle} — {tl}", fontsize=10, fontweight="bold")
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if mkey == "r2":
                ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    plt.tight_layout()
    _save(fig, "07_boxplot_cv.png")


# ── Gráfica 8: Scatter Predicted vs Actual ───────────────────────────────────
def plot_scatter_pred_vs_actual():
    n_m = len(MODEL_NAMES)
    n_t = len(TARGETS)
    fig, axes = plt.subplots(n_t, n_m, figsize=(4 * n_m, 4 * n_t), squeeze=False)
    fig.suptitle("Predicciones vs Valores Reales de Fatiga",
                 fontsize=13, fontweight="bold")

    for row, (target_name, _) in enumerate(TARGETS.items()):
        tl = target_name.replace("_", " ")
        for col, model_name in enumerate(MODEL_NAMES):
            ax    = axes[row, col]
            color = PALETTE.get(model_name, "#555")
            data  = all_preds_targets[target_name].get(model_name)
            if data is None:
                ax.set_visible(False)
                continue
            y_true, y_pred = data
            ax.scatter(y_true, y_pred, alpha=0.35, color=color, s=10, edgecolors="none")
            lim = [min(y_true.min(), y_pred.min()) - 2,
                   max(y_true.max(), y_pred.max()) + 2]
            ax.plot(lim, lim, "k--", linewidth=1.0, alpha=0.5)
            r2  = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            ax.text(0.05, 0.92, f"R²={r2:.3f}\nMAE={mae:.2f}",
                    transform=ax.transAxes, fontsize=7, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_xlabel("Valor Real", fontsize=8)
            ax.set_ylabel("Predicción", fontsize=8)
            ax.set_title(f"{model_name.replace('Custom ', '')} | {tl}", fontsize=8, fontweight="bold")
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save(fig, "08_scatter_pred_vs_actual.png")


# Ejecutar todas las gráficas
plot_comparativa_general()
plot_learning_curves()
plot_optuna_convergencia()
plot_optuna_histogramas()
plot_importancia_hp()
plot_radar_chart()
plot_boxplot_cv()
plot_scatter_pred_vs_actual()

print(f"\n[OK] Todas las gráficas guardadas en: {output_dir}")
print("\n══════════════════════════════════════════")
print(" EXPERIMENTO OPTUNA V2 — COMPLETADO")
print(f" DB SQLite : {db_path}")
print(f" Resultados: {output_dir}")
print("══════════════════════════════════════════")
