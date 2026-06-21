# -*- coding: utf-8 -*-
"""
Motor común de entrenamiento y validación para modelos de PyTorch en FatigueSet.
"""

import os
import time
import json
from typing import Dict, List, Optional, Tuple, Type
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..pipeline import FatigueSetPipeline
from .rnn import _prepare_target_table, _merge_raw_streams, _build_sequences, FatigueSequenceDataset


def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str
) -> float:
    model.train()
    total_loss = 0.0
    for xb, yb in dataloader:
        xb = xb.to(device)
        yb = yb.to(device)
        
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        
        # Gradient clipping para evitar inestabilidad en recurrentes
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0.0


def val_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            total_loss += loss.item()
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0.0


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    patience: int,
    checkpoint_path: str,
    device: str
) -> Dict[str, List[float]]:
    history = {"train_loss": [], "val_loss": []}
    best_val = float('inf')
    wait = 0
    
    # Guardar estado inicial como fallback
    torch.save(model.state_dict(), checkpoint_path)
    
    for epoch in range(1, epochs + 1):
        tr_loss = train_step(model, train_loader, loss_fn, optimizer, device)
        va_loss = val_step(model, val_loader, loss_fn, device)
        
        # Limpieza de nulos o inf
        if not np.isfinite(tr_loss):
            tr_loss = float('inf')
        if not np.isfinite(va_loss):
            va_loss = float('inf')
            
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        
        if va_loss < best_val:
            best_val = va_loss
            wait = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            wait += 1
            if wait >= patience:
                break
                
    return history


def train_kfold_cv(
    model_class: Type[nn.Module],
    model_kwargs: Dict,
    pipeline: Optional[FatigueSetPipeline] = None,
    dataset_path: str = 'fatigueset',
    window_size: int = 128,
    step: int = 32,
    seq_len: int = 128,
    lr: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 50,
    patience: int = 10,
    n_splits: int = 5,
    device: str = 'cpu',
    output_dir: str = 'models/rnn',
    seed: int = 42,
) -> Tuple[Dict[str, Dict[str, float]], float]:
    """Bucle genérico de K-Fold por participante."""
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    pipeline = pipeline or FatigueSetPipeline(dataset_path=dataset_path)
    
    raw = pipeline.cargar_dataset(verbose=False)
    df_ml = pipeline.construir_dataset_ml(raw)
    if df_ml is None or df_ml.empty:
        raise RuntimeError('No se pudo construir el dataframe ML para targets')
        
    df_targets = _prepare_target_table(df_ml)
    if df_targets.empty:
        raise RuntimeError('No hay targets válidos')
        
    df_raw = _merge_raw_streams(raw)
    
    # Si seq_len es None, usar window_size por compatibilidad
    if seq_len is None:
        seq_len = int(window_size)
        
    X, y, groups_arr, feature_columns = _build_sequences(
        df_raw=df_raw,
        df_targets=df_targets,
        seq_len=int(seq_len),
        step=int(step),
    )
    
    dataset = FatigueSequenceDataset(X, y)
    kf = GroupKFold(n_splits=n_splits)
    
    results: Dict[str, Dict[str, float]] = {}
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(dataset)), groups=groups_arr), start=1):
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        
        # Instanciar el modelo con la clase genérica
        model = model_class(input_size=len(feature_columns), **model_kwargs).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        ckpt = os.path.join(output_dir, f'model_fold_{fold}.pt')
        
        # Entrenar
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            loss_fn=criterion,
            optimizer=optimizer,
            epochs=epochs,
            patience=patience,
            checkpoint_path=ckpt,
            device=device
        )
        
        # Cargar el mejor checkpoint de este fold
        model.load_state_dict(torch.load(ckpt))
        model.eval()
        
        preds, ytrue = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                preds.append(model(xb).cpu().numpy())
                ytrue.append(yb.numpy())
                
        p = np.vstack(preds) if preds else np.empty((0, 2), dtype=np.float32)
        t = np.vstack(ytrue) if ytrue else np.empty((0, 2), dtype=np.float32)
        
        num_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        
        if len(t) == 0:
            fold_result = {
                'mse_fisica': float('nan'),
                'mae_fisica': float('nan'),
                'rmse_fisica': float('nan'),
                'r2_fisica': float('nan'),
                'mse_mental': float('nan'),
                'mae_mental': float('nan'),
                'rmse_mental': float('nan'),
                'r2_mental': float('nan'),
                'num_params': num_params,
                'n_test': 0,
            }
        else:
            fold_result = {
                'mse_fisica': float(mean_squared_error(t[:, 0], p[:, 0])),
                'mae_fisica': float(mean_absolute_error(t[:, 0], p[:, 0])),
                'rmse_fisica': float(np.sqrt(mean_squared_error(t[:, 0], p[:, 0]))),
                'r2_fisica': float(r2_score(t[:, 0], p[:, 0])),
                'mse_mental': float(mean_squared_error(t[:, 1], p[:, 1])),
                'mae_mental': float(mean_absolute_error(t[:, 1], p[:, 1])),
                'rmse_mental': float(np.sqrt(mean_squared_error(t[:, 1], p[:, 1]))),
                'r2_mental': float(r2_score(t[:, 1], p[:, 1])),
                'num_params': num_params,
                'n_test': int(len(test_idx)),
            }
            
        results[f'fold_{fold}'] = fold_result
        
    execution_time = time.time() - start_time
    
    # Guardar resultados JSON
    with open(os.path.join(output_dir, 'cv_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
        
    return results, execution_time
