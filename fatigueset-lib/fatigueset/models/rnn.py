"""RNN utilities for FatigueSet.

This module trains a vanilla Elman RNN over real temporal sequences built
from raw multimodal signals (chest + wrist streams) aligned by timestamp.
It also includes robust preprocessing to remove NaN/inf before training.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    Dataset = object
    Subset = None
    TensorDataset = None

from ..pipeline import FatigueSetPipeline


IDENTITY_COLS = {'participante', 'sesion', 'fase', 'intensidad', 'intensidad_num', 'fase_num'}


def _find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _to_datetime_col(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Find/convert a timestamp column and return (df, time_col)."""
    out = df.copy()
    time_col = _find_col(out, ('timestamp', 'time', 'datetime', 'fecha', 'ts'))
    if time_col is None:
        raise ValueError('No timestamp-like column found in raw signal dataframe')
    out[time_col] = pd.to_datetime(out[time_col], errors='coerce')
    out = out.dropna(subset=[time_col])
    return out, time_col


def _safe_numeric(df: pd.DataFrame, exclude: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Robust numeric cleanup: inf->nan, interpolate, ffill/bfill, fill 0."""
    out = df.copy()
    exclude_set = set(exclude or [])
    numeric_cols = [
        c for c in out.columns
        if c not in exclude_set and pd.api.types.is_numeric_dtype(out[c])
    ]
    if not numeric_cols:
        return out

    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    out[numeric_cols] = out[numeric_cols].interpolate(method='linear', limit_direction='both', axis=0)
    out[numeric_cols] = out[numeric_cols].ffill().bfill().fillna(0.0)
    return out


def _prefix_numeric_columns(df: pd.DataFrame, prefix: str, keep: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    keep_set = set(keep)
    rename_map = {}
    for c in out.columns:
        if c in keep_set:
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            rename_map[c] = f'{prefix}{c}'
    return out.rename(columns=rename_map)


def _prepare_target_table(df_ml: pd.DataFrame) -> pd.DataFrame:
    """Create one target row per (participante, sesion).

    Uses the last available phase (largest fase_num when present).
    """
    needed = {'participante', 'sesion', 'fatiga_fisica', 'fatiga_mental'}
    missing = needed - set(df_ml.columns)
    if missing:
        raise ValueError(f'Missing required target columns: {sorted(missing)}')

    t = df_ml.copy()
    if 'fase_num' in t.columns:
        t = t.sort_values(['participante', 'sesion', 'fase_num'])
    else:
        t = t.sort_values(['participante', 'sesion'])

    t = t.groupby(['participante', 'sesion'], as_index=False).tail(1)
    t = t[['participante', 'sesion', 'fatiga_fisica', 'fatiga_mental']].copy()
    t[['fatiga_fisica', 'fatiga_mental']] = t[['fatiga_fisica', 'fatiga_mental']].replace([np.inf, -np.inf], np.nan)
    t[['fatiga_fisica', 'fatiga_mental']] = t[['fatiga_fisica', 'fatiga_mental']].ffill().bfill()
    t = t.dropna(subset=['fatiga_fisica', 'fatiga_mental'])
    return t


def _merge_raw_streams(dataset_raw: Dict[str, object]) -> pd.DataFrame:
    """Build an aligned raw temporal table using chest + wrist streams.

    Strategy:
    - Use chest as base timeline per (participante, sesion)
    - As-of merge wrist streams by nearest previous timestamp
    - Keep only identity + timestamp + numeric sensor columns
    """
    chest = dataset_raw.get('chest')
    wrist = dataset_raw.get('wrist') or {}
    if chest is None or not isinstance(chest, pd.DataFrame) or chest.empty:
        raise ValueError('Raw chest dataframe not available; cannot build temporal sequences')

    chest = chest.copy()
    chest, chest_time = _to_datetime_col(chest)
    pid_col = _find_col(chest, ('participante', 'participant', 'subject', 'id'))
    ses_col = _find_col(chest, ('sesion', 'session'))
    if pid_col is None or ses_col is None:
        raise ValueError('Chest dataframe must include participante and sesion columns')

    base_keep = [pid_col, ses_col, chest_time]
    chest = _prefix_numeric_columns(chest, 'chest_', keep=base_keep)
    chest = _safe_numeric(chest, exclude=base_keep)
    chest = chest.sort_values([pid_col, ses_col, chest_time]).reset_index(drop=True)

    merged = chest

    # Merge each wrist stream if present
    if isinstance(wrist, dict):
        for stream_name, stream_df in wrist.items():
            if stream_df is None or not isinstance(stream_df, pd.DataFrame) or stream_df.empty:
                continue
            cur = stream_df.copy()
            cur, cur_time = _to_datetime_col(cur)
            cur_pid = _find_col(cur, ('participante', 'participant', 'subject', 'id'))
            cur_ses = _find_col(cur, ('sesion', 'session'))
            if cur_pid is None or cur_ses is None:
                continue

            keep_cols = [cur_pid, cur_ses, cur_time]
            cur = _prefix_numeric_columns(cur, f'wrist_{stream_name}_', keep=keep_cols)
            cur = _safe_numeric(cur, exclude=keep_cols)
            cur = cur.sort_values([cur_pid, cur_ses, cur_time]).reset_index(drop=True)

            # Normalize key names for merge
            cur = cur.rename(columns={cur_pid: pid_col, cur_ses: ses_col, cur_time: f'{stream_name}_time'})
            cur_time_norm = f'{stream_name}_time'
            cur = cur.sort_values([pid_col, ses_col, cur_time_norm]).reset_index(drop=True)

            # merge_asof can be fragile with global sorting across groups;
            # merge per (participante, sesion) to guarantee sorted keys.
            merged_parts: List[pd.DataFrame] = []
            value_cols = [
                c for c in cur.columns
                if c not in {pid_col, ses_col, cur_time_norm}
                and pd.api.types.is_numeric_dtype(cur[c])
            ]

            for (pval, sval), base_sub in merged.groupby([pid_col, ses_col], sort=False):
                base_sub = base_sub.sort_values(chest_time).copy()
                right_sub = cur[(cur[pid_col] == pval) & (cur[ses_col] == sval)].sort_values(cur_time_norm).copy()

                if right_sub.empty:
                    for c in value_cols:
                        base_sub[c] = np.nan
                    merged_parts.append(base_sub)
                    continue

                right_sub = right_sub.drop(columns=[pid_col, ses_col], errors='ignore')
                right_sub = right_sub[[cur_time_norm] + value_cols]

                msub = pd.merge_asof(
                    base_sub,
                    right_sub,
                    left_on=chest_time,
                    right_on=cur_time_norm,
                    direction='backward',
                    allow_exact_matches=True,
                )
                if cur_time_norm in msub.columns:
                    msub = msub.drop(columns=[cur_time_norm])
                merged_parts.append(msub)

            merged = pd.concat(merged_parts, ignore_index=True)

    merged = _safe_numeric(merged, exclude=[pid_col, ses_col, chest_time])
    merged = merged.rename(columns={pid_col: 'participante', ses_col: 'sesion', chest_time: 'timestamp'})
    return merged


def _build_sequences(
    df_raw: pd.DataFrame,
    df_targets: pd.DataFrame,
    seq_len: int,
    step: int,
    normalizar: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Create (X, y, groups, feature_names) from real raw temporal data."""
    feature_cols = [
        c for c in df_raw.columns
        if c not in {'participante', 'sesion', 'timestamp'}
        and pd.api.types.is_numeric_dtype(df_raw[c])
    ]
    if not feature_cols:
        raise ValueError('No numeric feature columns found in raw temporal dataframe')

    target_map = {
        (row['participante'], row['sesion']): np.array([row['fatiga_fisica'], row['fatiga_mental']], dtype=np.float32)
        for _, row in df_targets.iterrows()
    }

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    groups: List[str] = []

    grouped = df_raw.groupby(['participante', 'sesion'], dropna=False, sort=False)
    for (p, s), sub in grouped:
        if (p, s) not in target_map:
            continue
        sub = sub.sort_values('timestamp').reset_index(drop=True)
        
        # Copiar y opcionalmente aplicar Z-score por participante y sesión
        sub_features = sub[feature_cols].copy()
        if normalizar:
            for col in feature_cols:
                serie = pd.to_numeric(sub_features[col], errors='coerce')
                media = serie.mean()
                std = serie.std(ddof=0)
                sub_features[col] = 0.0 if pd.isna(std) or std == 0 else (serie - media) / std

        arr = sub_features.to_numpy(dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if len(arr) < seq_len:
            continue

        target = target_map[(p, s)]
        for i in range(0, len(arr) - seq_len + 1, step):
            X_list.append(arr[i:i + seq_len])
            y_list.append(target)
            groups.append(str(p))

    if not X_list:
        raise ValueError(
            'No raw sequences could be created. Try reducing seq_len/step or verify raw streams availability.'
        )

    X = np.stack(X_list)
    y = np.stack(y_list)
    g = np.array(groups)
    return X, y, g, feature_cols


class FatigueSequenceDataset(Dataset):
    """Simple tensor-backed dataset for RNN training."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


if nn is not None:
    class RNNFatiga(nn.Module):
        def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 1,
            dropout: float = 0.0,
            nonlinearity: str = 'tanh',
            bidirectional: bool = False,
            output_size: int = 2,
            batch_first: bool = True,
        ):
            super().__init__()
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity=nonlinearity,
                batch_first=batch_first,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            out_dim = hidden_size * (2 if bidirectional else 1)
            self.fc = nn.Linear(out_dim, output_size)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :])
else:
    class RNNFatiga:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('PyTorch is not available: install torch to use RNNFatiga')


def train_kfold(
    pipeline: Optional[FatigueSetPipeline] = None,
    dataset_path: str = 'fatigueset',
    window_size: int = 128,
    step: int = 32,
    seq_len: int = 128,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.2,
    lr: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 50,
    n_splits: int = 5,
    device: str = 'cpu',
    output_dir: str = 'output/rnn',
    seed: int = 42,
):
    """Train RNN using GroupKFold by participant over raw temporal sequences.

    Notes:
    - `seq_len` defines sequence length in raw timesteps.
    - `window_size` is kept for backward compatibility and is used when
      `seq_len` is not provided by callers.
    """
    if torch is None:
        raise RuntimeError('PyTorch is required to run the RNN training')

    if seq_len is None:
        seq_len = int(window_size)

    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    pipeline = pipeline or FatigueSetPipeline(dataset_path=dataset_path)

    print('Cargando dataset y construyendo dataframe ML...')
    raw = pipeline.cargar_dataset(verbose=False)
    df_ml = pipeline.construir_dataset_ml(raw)
    if df_ml is None or df_ml.empty:
        raise RuntimeError('No se pudo construir el dataframe ML para targets')

    df_targets = _prepare_target_table(df_ml)
    if df_targets.empty:
        raise RuntimeError('No hay targets válidos para entrenar la RNN')

    print('Construyendo secuencias temporales crudas...')
    df_raw = _merge_raw_streams(raw)
    X, y, groups_arr, feature_columns = _build_sequences(
        df_raw=df_raw,
        df_targets=df_targets,
        seq_len=int(seq_len),
        step=int(step),
    )

    dataset = FatigueSequenceDataset(X, y)

    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    kf = GroupKFold(n_splits=n_splits)
    results: Dict[str, Dict[str, float]] = {}

    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(dataset)), groups=groups_arr), start=1):
        print(f'Fold {fold}/{n_splits}: train {len(train_idx)} samples, test {len(test_idx)} samples')

        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        model = RNNFatiga(
            input_size=len(feature_columns),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        ckpt = os.path.join(output_dir, f'model_fold_{fold}.pt')
        torch.save(model.state_dict(), ckpt)
        best_val = float('inf')
        patience = 10
        wait = 0

        for epoch in range(1, epochs + 1):
            model.train()
            tr_losses: List[float] = []
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                tr_losses.append(float(loss.item()))

            model.eval()
            va_losses: List[float] = []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    out = model(xb)
                    va_losses.append(float(criterion(out, yb).item()))

            mean_tr = float(np.mean(tr_losses)) if tr_losses else float('inf')
            mean_va = float(np.mean(va_losses)) if va_losses else float('inf')
            if not np.isfinite(mean_tr):
                mean_tr = float('inf')
            if not np.isfinite(mean_va):
                mean_va = float('inf')

            print(f' epoch {epoch}: train_loss={mean_tr:.6f} val_loss={mean_va:.6f}')

            if mean_va < best_val:
                best_val = mean_va
                wait = 0
                torch.save(model.state_dict(), ckpt)
            else:
                wait += 1
                if wait >= patience:
                    print(' Early stopping')
                    break

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

        if len(t) == 0:
            fold_result = {
                'mse_fisica': float('nan'),
                'mae_fisica': float('nan'),
                'r2_fisica': float('nan'),
                'mse_mental': float('nan'),
                'mae_mental': float('nan'),
                'r2_mental': float('nan'),
                'n_test': 0,
            }
        else:
            fold_result = {
                'mse_fisica': float(mean_squared_error(t[:, 0], p[:, 0])),
                'mae_fisica': float(mean_absolute_error(t[:, 0], p[:, 0])),
                'r2_fisica': float(r2_score(t[:, 0], p[:, 0])),
                'mse_mental': float(mean_squared_error(t[:, 1], p[:, 1])),
                'mae_mental': float(mean_absolute_error(t[:, 1], p[:, 1])),
                'r2_mental': float(r2_score(t[:, 1], p[:, 1])),
                'n_test': int(len(test_idx)),
            }

        results[f'fold_{fold}'] = fold_result

    with open(os.path.join(output_dir, 'rnn_cv_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    return results


__all__ = ['FatigueSequenceDataset', 'RNNFatiga', 'train_kfold']
