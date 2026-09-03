import matplotlib
matplotlib.use('Agg')
# CONFIGURACIÓN GLOBAL: False = pruebas locales | True = servidor potente
SERVER_MODE = False

# SETUP e IMPORTACIONES
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os
import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Añadir fatigueset-lib al sys.path
lib_path = str(Path.cwd().parent / "fatigueset-lib")
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

from fatigueset import FatigueSetPipeline
from fatigueset.models import (
    FatigueSequenceDataset,
    RNNFatiga,
    CustomLSTMRegressor,
    CustomGRURegressor,
    CustomCNNLSTMRegressor,
    CustomTCNRegressor,
    CustomTSTransformerRegressor,
    CustomPatchTSTRegressor,
    CustomxLSTMRegressor,
)
from fatigueset.models.rnn import _prepare_target_table, _merge_raw_streams, _build_sequences
from fatigueset.models.optimizers import (
    build_optimizer,
    sample_model_hyperparams,
    sample_lstm,
    sample_gru,
    sample_cnn_lstm,
    sample_tcn,
    sample_transformer,
    sample_patchtst,
    sample_xlstm,
)

# Semilla global para reproducibilidad
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("[OK] Imports completados y path configurado.")
print(f"Dispositivo: {device}")

# ========================================

dataset_path = str(Path.cwd().parent / "fatigueset")
pipeline = FatigueSetPipeline(dataset_path=dataset_path, umbral_nulos=5.0)

print("Cargando dataset...")
raw = pipeline.cargar_dataset(verbose=False)

print("Preparando targets...")
df_ml = pipeline.construir_dataset_ml(raw)
df_targets = _prepare_target_table(df_ml)

print("Combinando streams fisiológicos...")
df_raw = _merge_raw_streams(raw)

# Parámetros de ventana temporal
SEQ_LEN = 128
STEP = 64

print("Construyendo secuencias...")
X_arr, y_arr, groups, feature_cols = _build_sequences(
    df_raw=df_raw,
    df_targets=df_targets,
    seq_len=SEQ_LEN,
    step=STEP,
)

dataset = FatigueSequenceDataset(X_arr, y_arr)
N_FEATURES = len(feature_cols)

print(f"[OK] Dimensiones:")
print(f"  - X: {X_arr.shape} (ventanas x seq_len x features)")
print(f"  - y: {y_arr.shape} (ventanas x 2)")
print(f"  - Grupos únicos: {len(np.unique(groups))}")

# ========================================

def instantiate_model(hyperparams: dict, n_features: int) -> nn.Module:
    """
    Instancia el modelo de PyTorch correspondiente a partir del diccionario
    de hiperparámetros devuelto por Optuna.
    """
    mt = hyperparams["model_type"]
    inp = n_features

    if mt == "LSTM":
        return CustomLSTMRegressor(
            input_size=inp,
            hidden_size=hyperparams["hidden_size"],
            num_layers=hyperparams["num_layers"],
            dropout=hyperparams["dropout"],
        )
    elif mt == "GRU":
        return CustomGRURegressor(
            input_size=inp,
            hidden_size=hyperparams["hidden_size"],
            num_layers=hyperparams["num_layers"],
            dropout=hyperparams["dropout"],
        )
    elif mt == "CNN-LSTM":
        return CustomCNNLSTMRegressor(
            input_size=inp,
            conv_channels=hyperparams["conv_channels"],
            kernel_size=hyperparams["kernel_size"],
            pool_size=hyperparams["pool_size"],
            hidden_size=hyperparams["hidden_size"],
            num_layers=hyperparams["num_layers"],
            dropout=hyperparams["dropout"],
        )
    elif mt == "TCN":
        return CustomTCNRegressor(
            input_size=inp,
            num_channels=hyperparams["num_channels"],
            kernel_size=hyperparams["kernel_size"],
            dropout=hyperparams["dropout"],
        )
    elif mt == "Transformer":
        return CustomTSTransformerRegressor(
            input_size=inp,
            d_model=hyperparams["d_model"],
            num_heads=hyperparams["num_heads"],
            num_layers=hyperparams["num_layers"],
            dim_feedforward=hyperparams["dim_feedforward"],
            dropout=hyperparams["dropout"],
        )
    elif mt == "PatchTST":
        return CustomPatchTSTRegressor(
            input_size=inp,
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
            input_size=inp,
            hidden_size=hyperparams["hidden_size"],
            num_layers=hyperparams["num_layers"],
            dropout=hyperparams["dropout"],
        )
    else:
        raise ValueError(f"Tipo de modelo desconocido: {mt}")


def quick_cv_eval(
    hyperparams: dict,
    dataset: FatigueSequenceDataset,
    groups: np.ndarray,
    n_features: int,
    n_splits: int = 3,
    epochs: int = 5,
    device: str = "cpu",
    trial=None,
) -> float:
    """
    Ejecuta una validación cruzada rápida GroupKFold y devuelve el MSE medio
    de validación. Usa el optimizador especificado en ``hyperparams``.

    Parámetros
    ----------
    hyperparams : dict
        Diccionario de hiperparámetros devuelto por una función ``sample_*``.
    trial : optuna.Trial, opcional
        Objeto Trial de Optuna para pruning (se puede pasar None en evaluación final).

    Retorna
    -------
    float
        MSE medio sobre todos los folds de validación.
    """
    batch_size = hyperparams.get("batch_size", 32)
    kf = GroupKFold(n_splits=n_splits)
    val_losses = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        kf.split(np.arange(len(dataset)), groups=groups)
    ):
        train_sub = Subset(dataset, train_idx)
        val_sub = Subset(dataset, val_idx)

        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)

        # Instanciar modelo con los hiperparámetros del trial
        model = instantiate_model(hyperparams, n_features).to(device)

        # Construir el optimizador desde el módulo customizado
        optimizer = build_optimizer(
            model=model,
            optimizer_name=hyperparams.get("optimizer", "Adam"),
            lr=hyperparams.get("lr", 1e-3),
            weight_decay=hyperparams.get("weight_decay", 0.0),
            momentum=hyperparams.get("momentum", 0.9),
            alpha=hyperparams.get("alpha", 0.99),
        )
        loss_fn = nn.MSELoss()

        # Bucle de entrenamiento
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        # Evaluación en validación
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                total_val += loss_fn(model(xb), yb).item()

        fold_mse = total_val / max(len(val_loader), 1)
        val_losses.append(fold_mse)

        # Pruning de Optuna: descartar trial si el fold actual no muestra potencial
        prune_trial = False
        if trial is not None:
            trial.report(np.mean(val_losses), step=fold_idx)
            if trial.should_prune():
                prune_trial = True

        # Limpiar memoria GPU explícitamente para evitar CUDA OOM
        del model, optimizer, train_loader, val_loader
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if prune_trial:
            raise optuna.exceptions.TrialPruned()

    return float(np.mean(val_losses))


print("[OK] Funciones de entrenamiento y evaluación definidas.")


# ========================================

# Configuración global del experimento Optuna
if 'SERVER_MODE' in globals() and SERVER_MODE:
    N_TRIALS = 25        # Más ensayos bayesianos en servidor (optimizado dentro de 12h)
    N_CV_SPLITS = 2      # 3-fold CV para evitar sobrepasar límites de tiempo
    EPOCHS_PER_TRIAL = 10# Más épocas por trial
else:
    N_TRIALS = 1
    N_CV_SPLITS = 2
    EPOCHS_PER_TRIAL = 1

# Definición de las familias de modelos con sus funciones de muestreo
MODEL_SAMPLERS = {
    "Custom LSTM":        sample_lstm,
    "Custom GRU":         sample_gru,
    "Custom CNN-LSTM":    sample_cnn_lstm,
    "Custom TCN":         sample_tcn,
    "Custom Transformer": sample_transformer,
    "Custom PatchTST":    sample_patchtst,
    "Custom xLSTM":       sample_xlstm,
}

# Almacenamiento de los mejores hiperparámetros por modelo
best_hyperparams_per_model = {}
optuna_results = []

for model_name, sampler_fn in MODEL_SAMPLERS.items():
    print(f"\n{'='*60}")
    print(f" Optimizando: {model_name}")
    print(f"{'='*60}")

    def make_objective(sampler_fn, ds, grps, n_feat, dev, ep, ns):
        def objective(trial):
            hyperparams = sampler_fn(trial)
            return quick_cv_eval(
                hyperparams=hyperparams,
                dataset=ds,
                groups=grps,
                n_features=n_feat,
                n_splits=ns,
                epochs=ep,
                device=dev,
                trial=trial,
            )
        return objective

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    objective_fn = make_objective(
        sampler_fn, dataset, groups, N_FEATURES, device, EPOCHS_PER_TRIAL, N_CV_SPLITS
    )

    t0 = time.time()
    study.optimize(objective_fn, n_trials=N_TRIALS, show_progress_bar=False)
    elapsed = time.time() - t0

    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

    print(f"  Trials completados: {n_complete} | Podados: {n_pruned}")
    print(f"  Mejor MSE validación: {study.best_value:.4f}")
    print(f"  Tiempo total: {elapsed:.1f}s")

    # Recuperar hiperparámetros óptimos
    best_trial = study.best_trial
    best_hp = sampler_fn(best_trial)
    best_hyperparams_per_model[model_name] = best_hp

    print(f"  Optimizador seleccionado: {best_hp.get('optimizer', 'N/A')}")
    print(f"  LR: {best_hp.get('lr', 'N/A'):.2e} | WD: {best_hp.get('weight_decay', 0):.2e}")

    optuna_results.append({
        "Modelo": model_name,
        "Mejor MSE (Optuna)": study.best_value,
        "Trials completados": n_complete,
        "Trials podados": n_pruned,
        "Optimizador": best_hp.get("optimizer", "N/A"),
        "LR": best_hp.get("lr", float("nan")),
        "Tiempo búsqueda (s)": elapsed,
    })

print("\n[OK] Búsqueda bayesiana completada para todos los modelos.")


# ========================================

df_optuna = pd.DataFrame(optuna_results)
print("\n=== RESULTADOS DE LA BÚSQUEDA BAYESIANA (Optuna) ===")
print(df_optuna.to_string(index=False))

# ========================================

# Configuración del experimento final
if 'SERVER_MODE' in globals() and SERVER_MODE:
    FINAL_EPOCHS = 30
    PATIENCE = 8
else:
    FINAL_EPOCHS = 1
    PATIENCE = 1

resultados_finales = []
kf_final = GroupKFold(n_splits=N_CV_SPLITS)

for model_name, best_hp in best_hyperparams_per_model.items():
    print(f"\nEvaluando {model_name} con hiperparámetros óptimos...")
    t0 = time.time()

    # Contar parámetros
    tmp_model = instantiate_model(best_hp, N_FEATURES)
    num_params = sum(p.numel() for p in tmp_model.parameters() if p.requires_grad)
    del tmp_model

    batch_size = best_hp.get("batch_size", 32)
    fold_r2, fold_mae, fold_rmse = [], [], []

    for train_idx, val_idx in kf_final.split(np.arange(len(dataset)), groups=groups):
        train_sub = Subset(dataset, train_idx)
        val_sub = Subset(dataset, val_idx)

        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)

        model = instantiate_model(best_hp, N_FEATURES).to(device)

        # Usar el optimizador óptimo encontrado por Optuna
        optimizer = build_optimizer(
            model=model,
            optimizer_name=best_hp.get("optimizer", "Adam"),
            lr=best_hp.get("lr", 1e-3),
            weight_decay=best_hp.get("weight_decay", 0.0),
            momentum=best_hp.get("momentum", 0.9),
            alpha=best_hp.get("alpha", 0.99),
        )
        loss_fn = nn.MSELoss()

        # Entrenamiento con early stopping
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(FINAL_EPOCHS):
            # Entrenamiento
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            # Early stopping
            model.eval()
            val_loss_ep = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss_ep += loss_fn(model(xb), yb).item()
            val_loss_ep /= max(len(val_loader), 1)

            if val_loss_ep < best_val_loss:
                best_val_loss = val_loss_ep
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break

        # Evaluación de métricas finales
        model.eval()
        preds, targets_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds.append(model(xb).cpu().numpy())
                targets_true.append(yb.numpy())

        p_arr = np.vstack(preds)
        t_arr = np.vstack(targets_true)

        fold_mae.append(mean_absolute_error(t_arr, p_arr))
        fold_rmse.append(np.sqrt(mean_squared_error(t_arr, p_arr)))
        fold_r2.append(r2_score(t_arr, p_arr))

        # Limpiar memoria GPU explícitamente
        del model, optimizer, train_loader, val_loader
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    resultados_finales.append({
        "Modelo": model_name,
        "Optimizador": best_hp.get("optimizer", "N/A"),
        "LR": best_hp.get("lr", float("nan")),
        "MAE Medio": float(np.mean(fold_mae)),
        "RMSE Medio": float(np.mean(fold_rmse)),
        "R² Medio": float(np.mean(fold_r2)),
        "Nº Parámetros": num_params,
        "Tiempo CV (s)": float(elapsed),
    })
    print(f"  [OK] MAE: {np.mean(fold_mae):.4f} | R²: {np.mean(fold_r2):.4f} "
          f"| Optimizer: {best_hp.get('optimizer', 'N/A')} | Params: {num_params:,}")

df_final = pd.DataFrame(resultados_finales)
print("\n=== TABLA COMPARATIVA FINAL (Hiperparámetros Óptimos + Optimizadores Personalizados) ===")
print(df_final.to_string(index=False))


# ========================================

# Configuración del baseline con Adam fijo (mismo número de épocas)
BASELINE_CONFIG = {
    "Custom LSTM":        lambda: CustomLSTMRegressor(input_size=N_FEATURES, hidden_size=64, num_layers=2, dropout=0.2),
    "Custom GRU":         lambda: CustomGRURegressor(input_size=N_FEATURES, hidden_size=64, num_layers=2, dropout=0.2),
    "Custom CNN-LSTM":    lambda: CustomCNNLSTMRegressor(input_size=N_FEATURES, conv_channels=64, hidden_size=64, num_layers=2),
    "Custom TCN":         lambda: CustomTCNRegressor(input_size=N_FEATURES, num_channels=[64, 64, 128]),
    "Custom Transformer": lambda: CustomTSTransformerRegressor(input_size=N_FEATURES, d_model=64, num_heads=4, num_layers=2),
    "Custom PatchTST":    lambda: CustomPatchTSTRegressor(input_size=N_FEATURES, patch_len=16, stride=8, d_model=64, num_heads=4, num_layers=2),
    "Custom xLSTM":       lambda: CustomxLSTMRegressor(input_size=N_FEATURES, hidden_size=64, num_layers=2),
}

resultados_baseline = []
kf_base = GroupKFold(n_splits=N_CV_SPLITS)

for model_name, model_fn in BASELINE_CONFIG.items():
    print(f"  Baseline (Adam): {model_name}...")
    fold_r2, fold_mae, fold_rmse = [], [], []

    for train_idx, val_idx in kf_base.split(np.arange(len(dataset)), groups=groups):
        train_sub = Subset(dataset, train_idx)
        val_sub = Subset(dataset, val_idx)

        train_loader = DataLoader(train_sub, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=32, shuffle=False)

        model = model_fn().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        for epoch in range(FINAL_EPOCHS):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        model.eval()
        preds, targets_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds.append(model(xb).cpu().numpy())
                targets_true.append(yb.numpy())

        p_arr = np.vstack(preds)
        t_arr = np.vstack(targets_true)
        fold_mae.append(mean_absolute_error(t_arr, p_arr))
        fold_rmse.append(np.sqrt(mean_squared_error(t_arr, p_arr)))
        fold_r2.append(r2_score(t_arr, p_arr))

        # Limpiar memoria GPU explícitamente
        del model, optimizer, train_loader, val_loader
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    resultados_baseline.append({
        "Modelo": model_name,
        "MAE (Adam)": float(np.mean(fold_mae)),
        "RMSE (Adam)": float(np.mean(fold_rmse)),
        "R² (Adam)": float(np.mean(fold_r2)),
    })

df_baseline = pd.DataFrame(resultados_baseline)

# Combinación de tablas para comparación
df_comp = df_final[["Modelo", "Optimizador", "MAE Medio", "R² Medio"]].merge(
    df_baseline[["Modelo", "MAE (Adam)", "R² (Adam)"]], on="Modelo"
)
df_comp["ΔMAE"] = df_comp["MAE (Adam)"] - df_comp["MAE Medio"]
df_comp["ΔR²"] = df_comp["R² Medio"] - df_comp["R² (Adam)"]

print("\n=== COMPARATIVA: Adam Fijo vs. Optimizador Óptimo (Optuna) ===")
print(df_comp.to_string(index=False))
print("\n(ΔMAE positivo = mejora | ΔR² positivo = mejora)")


# ========================================

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle(
    "Optimización Bayesiana con Optuna y Optimizadores Personalizados\n"
    "Comparativa de Rendimiento en FatigueSet",
    fontsize=14, fontweight='bold'
)

models_names = df_comp["Modelo"].str.replace("Custom ", "", regex=False)

# --- Gráfica 1: MAE Comparativo ---
ax1 = axes[0]
x = np.arange(len(models_names))
w = 0.35
bars1 = ax1.bar(x - w/2, df_comp["MAE (Adam)"], w, label="Adam (baseline)", color="#4c72b0", alpha=0.85)
bars2 = ax1.bar(x + w/2, df_comp["MAE Medio"], w, label="Optimizador óptimo (Optuna)", color="#dd8452", alpha=0.85)
ax1.set_xticks(x)
ax1.set_xticklabels(models_names, rotation=30, ha='right', fontsize=9)
ax1.set_ylabel("MAE", fontsize=11)
ax1.set_title("MAE: Adam vs. Optimizador Óptimo", fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# --- Gráfica 2: Distribución de optimizadores ---
ax2 = axes[1]
opt_counts = df_final["Optimizador"].value_counts()
colors_pie = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
ax2.pie(
    opt_counts.values,
    labels=opt_counts.index,
    autopct='%1.0f%%',
    colors=colors_pie[:len(opt_counts)],
    startangle=90,
    textprops={'fontsize': 10}
)
ax2.set_title("Distribución de Optimizadores\nSeleccionados por Optuna", fontsize=11)

# --- Gráfica 3: Mejora relativa de R² ---
ax3 = axes[2]
delta_r2 = df_comp["ΔR²"].values
bar_colors = ["#55a868" if v >= 0 else "#c44e52" for v in delta_r2]
ax3.barh(models_names, delta_r2, color=bar_colors, alpha=0.85)
ax3.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax3.set_xlabel("ΔR² (Optuna − Adam)", fontsize=11)
ax3.set_title("Mejora de R² con\nOptimizador Óptimo", fontsize=11)
ax3.grid(axis='x', alpha=0.3)

plt.tight_layout()
output_dir = Path.cwd().parent / "output"
output_dir.mkdir(exist_ok=True)
fig_path = output_dir / "optuna_optimizadores_personalizados.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
# plt.show()
print(f"[OK] Figura guardada en: {fig_path}")

# ========================================

output_dir = Path.cwd().parent / "output"
output_dir.mkdir(exist_ok=True)

# Guardar tabla final como CSV
csv_path = output_dir / "comparativa_optuna_optimizadores.csv"
df_final.to_csv(csv_path, index=False)
print(f"Tabla comparativa guardada en: {csv_path}")

# Guardar tabla de búsqueda Optuna
csv_optuna_path = output_dir / "resultados_busqueda_optuna.csv"
df_optuna.to_csv(csv_optuna_path, index=False)
print(f"Resultados Optuna guardados en: {csv_optuna_path}")

# Guardar mejores hiperparámetros como JSON (para reproducibilidad)
# Convertir valores de numpy a tipos Python nativos para serialización JSON
def to_python_native(obj):
    if isinstance(obj, dict):
        return {k: to_python_native(v) for k, v in obj.items()}
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, list):
        return [to_python_native(v) for v in obj]
    else:
        return obj

json_path = output_dir / "mejores_hiperparametros_optuna.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(to_python_native(best_hyperparams_per_model), f, indent=2, ensure_ascii=False)
print(f"Mejores hiperparámetros guardados en: {json_path}")

print("\n[OK] Resultados persistidos correctamente.")

# ========================================

print("\n=== MEJORES HIPERPARÁMETROS POR MODELO ===")
for model_name, hp in best_hyperparams_per_model.items():
    print(f"\n► {model_name}")
    # Separar parámetros de arquitectura y de optimizador
    arch_keys = [k for k in hp if k not in ('optimizer', 'lr', 'weight_decay', 'momentum', 'alpha', 'batch_size', 'model_type')]
    opt_keys = [k for k in hp if k in ('optimizer', 'lr', 'weight_decay', 'momentum', 'alpha')]
    
    print(f"  Arquitectura: batch={hp.get('batch_size', 'N/A')}")
    for k in arch_keys:
        print(f"    {k}: {hp[k]}")
    print(f"  Optimizador:")
    for k in opt_keys:
        v = hp[k]
        if isinstance(v, float):
            print(f"    {k}: {v:.2e}")
        else:
            print(f"    {k}: {v}")

# ========================================

