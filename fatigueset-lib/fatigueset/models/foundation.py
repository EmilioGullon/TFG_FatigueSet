# -*- coding: utf-8 -*-
"""
Módulo de integración de Modelos Fundacionales Preentrenados para FatigueSet.

Implementa dos modelos del survey "Foundation Models for Time Series: A Survey" (2025):
  - MOMENT (AutonLab/MOMENT-1-*): Modelo encoder-only basado en parches. Se adapta
    para regresión de fatiga mediante fine-tuning de una cabeza lineal, manteniendo
    el backbone preentrenado congelado (o con LoRA opcional en el servidor).
  - Chronos (amazon/chronos-t5-*): Framework probabilístico decoder-only basado en T5.
    Se usa en modo zero-shot como baseline probabilístico. Dado que Chronos predice
    distribuciones de la siguiente observación de una serie temporal, se adapta a
    FatigueSet tratando la secuencia de puntuaciones de fatiga como una serie a predecir.

Uso local (pruebas):
  MOMENT_CHECKPOINT = "AutonLab/MOMENT-1-small"   # 40M parámetros
  CHRONOS_CHECKPOINT = "amazon/chronos-t5-tiny"    # 8M parámetros

Uso en servidor (experimento completo):
  MOMENT_CHECKPOINT = "AutonLab/MOMENT-1-large"    # 385M parámetros
  CHRONOS_CHECKPOINT = "amazon/chronos-t5-large"   # 710M parámetros

Referencia:
  Goswami et al. (2024). MOMENT: A Family of Open Time-series Foundation Models. ICML.
  Ansari et al. (2024). Chronos: Learning the Language of Time Series. arXiv:2403.07815.
  Kottapalli et al. (2025). Foundation Models for Time Series: A Survey. arXiv.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset


# ---------------------------------------------------------------------------
# Configuración de checkpoints: cambiar para el servidor
# ---------------------------------------------------------------------------

MOMENT_LOCAL = "AutonLab/MOMENT-1-small"    # 40 M parámetros (pruebas locales)
MOMENT_SERVER = "AutonLab/MOMENT-1-large"   # 385 M parámetros (servidor)

CHRONOS_LOCAL = "amazon/chronos-t5-tiny"    # 8 M parámetros (pruebas locales)
CHRONOS_SERVER = "amazon/chronos-t5-large"  # 710 M parámetros (servidor)


# ---------------------------------------------------------------------------
# 1. MOMENT: Regresor de Fatiga con Fine-Tuning
# ---------------------------------------------------------------------------

class MOMENTFatigueRegressor(nn.Module):
    """
    Regresor de fatiga construido sobre el backbone MOMENT preentrenado.

    Arquitectura:
        - Backbone MOMENT (encoder Transformer por parches, congelado por defecto).
        - Cabeza de regresión lineal: ``Linear(d_model, output_size)``.

    El backbone actúa como extractor de características universales de series temporales,
    entrenado sobre 1.130 millones de puntos de datos de múltiples dominios. La cabeza
    de regresión se adapta específicamente a la tarea de predicción de fatiga en FatigueSet
    mediante un fine-tuning ligero sobre los datos etiquetados disponibles.

    Parámetros
    ----------
    checkpoint : str
        Identificador de HuggingFace del modelo MOMENT. Por defecto, la versión pequeña
        para pruebas locales.
    n_channels : int
        Número de canales de entrada (variables fisiológicas). Default: 23.
    seq_len : int
        Longitud de la secuencia temporal de entrada. Debe coincidir con la longitud
        de las ventanas construidas en el pipeline de FatigueSet. Default: 512.
    output_size : int
        Número de salidas de regresión. Default: 2 (fatiga_fisica, fatiga_mental).
    freeze_backbone : bool
        Si es True, congela todos los parámetros del backbone durante el fine-tuning,
        entrenando únicamente la cabeza de regresión (más eficiente, menos riesgo
        de sobreajuste en datasets pequeños). Default: True.
    dropout : float
        Dropout aplicado antes de la capa de regresión final. Default: 0.1.

    Notas
    -----
    MOMENT espera entrada de forma ``(batch, n_channels, seq_len)`` (canales en segunda
    dimensión), a diferencia de nuestros modelos propios que usan ``(batch, seq_len,
    n_channels)``. La función ``forward`` realiza la transposición internamente.
    """

    def __init__(
        self,
        checkpoint: str = MOMENT_LOCAL,
        n_channels: int = 23,
        seq_len: int = 512,
        output_size: int = 2,
        freeze_backbone: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.output_size = output_size
        self.freeze_backbone = freeze_backbone

        self._backbone = None
        self._d_model = None
        self._backbone_loaded = False

        self.dropout_layer = nn.Dropout(dropout)

        # La cabeza de regresión se construirá al cargar el backbone
        self.regression_head: Optional[nn.Linear] = None

    def _load_backbone(self) -> None:
        """
        Carga el backbone MOMENT desde HuggingFace de forma perezosa (lazy loading).
        Se llama automáticamente en el primer forward pass o explícitamente.
        """
        if self._backbone_loaded:
            return

        try:
            from momentfm import MOMENTPipeline
        except ImportError as e:
            raise ImportError(
                "El paquete 'momentfm' no está instalado. "
                "Instálalo con: pip install momentfm"
            ) from e

        print(f"[MOMENT] Cargando backbone desde '{self.checkpoint}'...")
        t0 = time.time()

        # Cargar MOMENT en modo clasificación (reutilizamos el encoder)
        # task_name='classification' proporciona embeddings de la secuencia completa
        moment_pipe = MOMENTPipeline.from_pretrained(
            self.checkpoint,
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.n_channels,
                "seq_len": self.seq_len,
                "num_class": self.output_size,  # placeholder, se sustituirá por cabeza propia
            }
        )

        # Usar la tubería completa como backbone
        self._backbone = moment_pipe

        if hasattr(self._backbone, "config") and hasattr(self._backbone.config, "d_model"):
            self._d_model = self._backbone.config.d_model
        elif hasattr(self._backbone, "encoder") and hasattr(self._backbone.encoder, "config"):
            self._d_model = self._backbone.encoder.config.d_model
        else:
            self._d_model = getattr(self._backbone, "d_model", 768)

        # Construir cabeza de regresión propia (más flexible que la clasificación interna)
        self.regression_head = nn.Linear(self._d_model, self.output_size)
        nn.init.xavier_uniform_(self.regression_head.weight)
        nn.init.zeros_(self.regression_head.bias)

        # Congelar backbone si se solicita (fine-tuning de la cabeza únicamente)
        if self.freeze_backbone:
            for param in self._backbone.parameters():
                param.requires_grad = False
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            print(
                f"[MOMENT] Backbone congelado. "
                f"Parámetros entrenables: {trainable:,} / {total:,} "
                f"({100*trainable/total:.1f}%)"
            )
        else:
            total = sum(p.numel() for p in self.parameters())
            print(f"[MOMENT] Backbone descongelado. Parámetros totales: {total:,}")

        self._backbone_loaded = True
        print(f"[MOMENT] Backbone cargado en {time.time() - t0:.1f}s")

    def load_backbone(self) -> "MOMENTFatigueRegressor":
        """Carga explícita del backbone. Útil para pre-calentar antes del entrenamiento."""
        self._load_backbone()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paso forward: codifica la secuencia fisiológica y produce la predicción de fatiga.

        Parámetros
        ----------
        x : torch.Tensor
            Secuencia de entrada con forma ``(batch, seq_len, n_channels)`` — mismo
            formato que el resto de modelos de FatigueSet. Se transpone internamente
            al formato MOMENT: ``(batch, n_channels, seq_len)``.

        Retorna
        -------
        torch.Tensor
            Predicciones de forma ``(batch, output_size)``.
        """
        if not self._backbone_loaded:
            self._load_backbone()

        # Transposición: (B, seq_len, n_channels) → (B, n_channels, seq_len)
        # Hacemos el tensor contiguo para evitar errores en las operaciones de view internas de MOMENT
        x_moment = x.transpose(1, 2).contiguous()

        # Obtener embeddings usando el método embed de la tubería
        # outputs.embeddings: (B, d_model)
        outputs = self._backbone.embed(x_enc=x_moment, reduction="mean")
        pooled = outputs.embeddings  # (B, d_model)

        return self.regression_head(self.dropout_layer(pooled))

    def count_parameters(self) -> Dict[str, int]:
        """Retorna un diccionario con el conteo de parámetros totales y entrenables."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ---------------------------------------------------------------------------
# 2. Evaluador Zero-Shot de Chronos (predicción probabilística)
# ---------------------------------------------------------------------------

class ChronosZeroShotEvaluator:
    """
    Evaluador zero-shot basado en Chronos para la predicción probabilística de fatiga.

    Dado que Chronos está preentrenado para **forecasting** (predecir el siguiente
    valor de una serie temporal), se adapta a FatigueSet de la siguiente forma:
      1. Cada canal fisiológico de la ventana de entrada se trata como una serie
         univariada independiente.
      2. Chronos predice la distribución del siguiente paso de cada canal.
      3. Las medianas de predicción se usan como características para una **regresión
         lineal de sonda** (linear probe) que mapea al espacio de fatiga.

    Esta estrategia es coherente con el paradigma descrito en el survey (Sección 7.5
    sobre "LLMs para series temporales"): el LLM/Foundation Model actúa como
    extractor de características universales, y la cabeza downstream es ligera.

    Parámetros
    ----------
    checkpoint : str
        Identificador HuggingFace del modelo Chronos. Default: versión tiny para local.
    prediction_length : int
        Número de pasos futuros que Chronos predice por canal. Default: 1.
    num_samples : int
        Número de muestras de Monte Carlo para estimar la distribución predictiva.
        Más muestras → mejor estimación de CRPS pero más lento. Default: 20 (local),
        500 en servidor.
    device : str
        Dispositivo de inferencia. Default: 'cpu'.
    """

    def __init__(
        self,
        checkpoint: str = CHRONOS_LOCAL,
        prediction_length: int = 1,
        num_samples: int = 20,
        device: str = "cpu",
    ):
        self.checkpoint = checkpoint
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.device = device
        self._pipeline = None

    def _load_pipeline(self) -> None:
        """Carga el pipeline de Chronos de forma perezosa."""
        if self._pipeline is not None:
            return

        try:
            from chronos import ChronosPipeline
        except ImportError as e:
            raise ImportError(
                "El paquete 'chronos-forecasting' no está instalado. "
                "Instálalo con: pip install chronos-forecasting"
            ) from e

        print(f"[Chronos] Cargando pipeline desde '{self.checkpoint}'...")
        t0 = time.time()
        self._pipeline = ChronosPipeline.from_pretrained(
            self.checkpoint,
            device_map=self.device,
            torch_dtype=torch.float32,
        )
        print(f"[Chronos] Pipeline cargado en {time.time() - t0:.1f}s")

    def extract_features(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Extrae características de cada muestra usando Chronos como codificador.

        Para cada ventana ``X[i]`` de forma ``(seq_len, n_channels)``, se predice
        la mediana del siguiente punto para cada canal. Las medianas de todos los
        canales se concatenan como vector de características.

        Parámetros
        ----------
        X : np.ndarray
            Array de secuencias, forma ``(N, seq_len, n_channels)``.
        batch_size : int
            Número de secuencias procesadas simultáneamente.

        Retorna
        -------
        np.ndarray
            Matriz de características, forma ``(N, n_channels)``.
        """
        self._load_pipeline()

        N, seq_len, n_channels = X.shape
        features = np.zeros((N, n_channels), dtype=np.float32)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = X[start:end]  # (B, seq_len, n_channels)

            batch_features = np.zeros((end - start, n_channels), dtype=np.float32)

            for ch in range(n_channels):
                # Extraer el canal ch para todas las muestras del batch
                channel_series = [
                    torch.tensor(batch[i, :, ch], dtype=torch.float32)
                    for i in range(end - start)
                ]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Chronos predice distribución del siguiente punto
                    quantiles, mean = self._pipeline.predict_quantiles(
                        inputs=channel_series,
                        prediction_length=self.prediction_length,
                        quantile_levels=[0.1, 0.5, 0.9],
                        num_samples=self.num_samples,
                    )

                # Mediana (quantil 0.5) del siguiente paso como feature
                # quantiles: (B, prediction_length, 3)
                median_pred = quantiles[:, 0, 1].numpy()  # (B,)
                batch_features[:, ch] = median_pred

            features[start:end] = batch_features

        return features

    def fit_probe(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
    ) -> Tuple[np.ndarray, Any]:
        """
        Entrena una regresión lineal de sonda sobre las características de Chronos.

        Parámetros
        ----------
        X_train, X_val : np.ndarray
            Arrays de secuencias, forma ``(N, seq_len, n_channels)``.
        y_train : np.ndarray
            Etiquetas de fatiga, forma ``(N, 2)``.

        Retorna
        -------
        Tuple[np.ndarray, LinearRegression]
            Predicciones en X_val y el modelo de sonda entrenado.
        """
        from sklearn.linear_model import Ridge

        print("[Chronos] Extrayendo características del conjunto de entrenamiento...")
        feats_train = self.extract_features(X_train)
        print("[Chronos] Extrayendo características del conjunto de validación...")
        feats_val = self.extract_features(X_val)

        probe = Ridge(alpha=1.0)
        probe.fit(feats_train, y_train)
        y_pred = probe.predict(feats_val)

        return y_pred, probe

    def predict_quantiles_for_crps(
        self,
        X: np.ndarray,
        quantile_levels: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Predice cuantiles de la distribución predictiva de Chronos para el cálculo
        de CRPS (Continuous Ranked Probability Score).

        Parámetros
        ----------
        X : np.ndarray
            Array de secuencias, forma ``(N, seq_len, n_channels)``.
        quantile_levels : list of float, opcional
            Cuantiles a predecir. Default: [0.1, 0.2, ..., 0.9].

        Retorna
        -------
        np.ndarray
            Cuantiles de forma ``(N, n_quantiles, n_channels)``.
        """
        self._load_pipeline()

        if quantile_levels is None:
            quantile_levels = [round(q * 0.1, 1) for q in range(1, 10)]

        N, seq_len, n_channels = X.shape
        all_quantiles = np.zeros((N, len(quantile_levels), n_channels), dtype=np.float32)

        for ch in range(n_channels):
            channel_series = [
                torch.tensor(X[i, :, ch], dtype=torch.float32) for i in range(N)
            ]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                quantiles, _ = self._pipeline.predict_quantiles(
                    inputs=channel_series,
                    prediction_length=self.prediction_length,
                    quantile_levels=quantile_levels,
                    num_samples=self.num_samples,
                )
            # quantiles: (N, prediction_length, n_quantiles)
            all_quantiles[:, :, ch] = quantiles[:, 0, :].numpy()

        return all_quantiles


# ---------------------------------------------------------------------------
# 3. Funciones de Métricas Probabilísticas
# ---------------------------------------------------------------------------

def compute_crps_gaussian(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """
    Calcula el CRPS (Continuous Ranked Probability Score) bajo el supuesto
    de distribución Normal para las predicciones.

    El CRPS generaliza el MAE al espacio probabilístico: un CRPS de 0 indica
    predicción perfecta, y para predicciones deterministas degeneración el CRPS
    al MAE estándar (Gneiting & Raftery, 2007).

    Parámetros
    ----------
    y_true : np.ndarray, forma (N,)
        Valores reales observados.
    mu : np.ndarray, forma (N,)
        Media predictiva de la distribución Normal.
    sigma : np.ndarray, forma (N,)
        Desviación estándar predictiva. Debe ser > 0.

    Retorna
    -------
    float
        CRPS medio sobre todas las muestras.

    Referencia
    ----------
    Gneiting, T., & Raftery, A. E. (2007). Strictly Proper Scoring Rules,
    Prediction, and Estimation. JASA, 102(477), 359–378.
    """
    from scipy.stats import norm

    sigma = np.maximum(sigma, 1e-8)  # Evitar división por cero
    z = (y_true - mu) / sigma
    crps_values = sigma * (
        z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)
    )
    return float(np.mean(crps_values))


def compute_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Calcula la cobertura empírica de un intervalo de predicción probabilístico.

    Parámetros
    ----------
    y_true : np.ndarray, forma (N,)
        Valores reales observados.
    lower : np.ndarray, forma (N,)
        Límite inferior del intervalo de predicción.
    upper : np.ndarray, forma (N,)
        Límite superior del intervalo de predicción.

    Retorna
    -------
    float
        Fracción de observaciones dentro del intervalo [lower, upper].
        Un intervalo de 90% bien calibrado debería tener cobertura ≈ 0.90.
    """
    within = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(within))


def evaluate_probabilistic_metrics(
    y_true: np.ndarray,
    y_pred_samples: np.ndarray,
    ci_level: float = 0.90,
) -> Dict[str, float]:
    """
    Evalúa métricas probabilísticas completas dado un conjunto de muestras predictivas.

    Parámetros
    ----------
    y_true : np.ndarray, forma (N, 2)
        Valores reales de fatiga_fisica ([:, 0]) y fatiga_mental ([:, 1]).
    y_pred_samples : np.ndarray, forma (N, n_samples, 2)
        Muestras de la distribución predictiva por muestra y por dimensión.
    ci_level : float
        Nivel del intervalo de confianza. Default: 0.90 (90%).

    Retorna
    -------
    Dict[str, float]
        Diccionario con MAE, RMSE, R², CRPS y cobertura para cada dimensión.
    """
    alpha = (1 - ci_level) / 2
    lower_q = alpha
    upper_q = 1 - alpha

    metrics = {}
    dim_names = ["fisica", "mental"]

    for dim, name in enumerate(dim_names):
        yt = y_true[:, dim]
        samples_dim = y_pred_samples[:, :, dim]  # (N, n_samples)

        mu = samples_dim.mean(axis=1)
        sigma = samples_dim.std(axis=1)
        lower = np.quantile(samples_dim, lower_q, axis=1)
        upper = np.quantile(samples_dim, upper_q, axis=1)

        metrics[f"mae_{name}"] = float(mean_absolute_error(yt, mu))
        metrics[f"rmse_{name}"] = float(np.sqrt(mean_squared_error(yt, mu)))
        metrics[f"r2_{name}"] = float(r2_score(yt, mu))
        metrics[f"crps_{name}"] = compute_crps_gaussian(yt, mu, sigma)
        metrics[f"coverage_{ci_level:.0%}_{name}"] = compute_coverage(yt, lower, upper)

    return metrics


# ---------------------------------------------------------------------------
# 4. Bucle de Fine-Tuning de MOMENT con GroupKFold
# ---------------------------------------------------------------------------

def finetune_moment_kfold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    checkpoint: str = MOMENT_LOCAL,
    seq_len: int = 512,
    n_channels: int = 23,
    freeze_backbone: bool = True,
    lr: float = 1e-4,
    epochs: int = 10,
    patience: int = 5,
    batch_size: int = 16,
    dropout: float = 0.1,
    n_splits: int = 3,
    device: str = "cpu",
) -> Tuple[List[Dict[str, float]], float]:
    """
    Fine-tuning de MOMENT con validación cruzada GroupKFold por participante.

    Congela el backbone MOMENT y entrena únicamente la cabeza de regresión lineal
    sobre las representaciones preentrenadas (transfer learning).

    Parámetros
    ----------
    X : np.ndarray, forma (N, seq_len, n_channels)
        Secuencias de entrada.
    y : np.ndarray, forma (N, 2)
        Etiquetas de fatiga.
    groups : np.ndarray, forma (N,)
        Grupos de participante para GroupKFold.
    checkpoint : str
        Checkpoint HuggingFace de MOMENT.
    seq_len : int
        Longitud de secuencia.
    n_channels : int
        Número de canales de entrada.
    freeze_backbone : bool
        Si True, solo se entrena la cabeza de regresión.
    lr : float
        Tasa de aprendizaje.
    epochs : int
        Número máximo de épocas de fine-tuning.
    patience : int
        Paciencia para early stopping.
    batch_size : int
        Tamaño de batch.
    dropout : float
        Dropout en la cabeza.
    n_splits : int
        Número de folds GroupKFold.
    device : str
        Dispositivo de cómputo.

    Retorna
    -------
    Tuple[List[Dict], float]
        Lista de métricas por fold y tiempo total de entrenamiento.
    """
    from torch.utils.data import TensorDataset

    kf = GroupKFold(n_splits=n_splits)
    fold_results = []
    t_start = time.time()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    full_dataset = TensorDataset(X_tensor, y_tensor)

    for fold_idx, (train_idx, val_idx) in enumerate(
        kf.split(np.arange(len(X)), groups=groups), start=1
    ):
        print(f"\n[MOMENT] Fold {fold_idx}/{n_splits} — train: {len(train_idx)}, val: {len(val_idx)}")

        train_sub = Subset(full_dataset, train_idx)
        val_sub = Subset(full_dataset, val_idx)
        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, num_workers=0)

        # Instanciar modelo fresco por fold para evitar contaminación
        model = MOMENTFatigueRegressor(
            checkpoint=checkpoint,
            n_channels=n_channels,
            seq_len=seq_len,
            output_size=2,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
        )
        model.load_backbone()
        model = model.to(device)

        # Solo optimizar los parámetros entrenables (cabeza + capas de normalización)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=1e-4,
        )
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            # Entrenamiento
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= max(len(train_loader), 1)

            # Validación
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss += loss_fn(model(xb), yb).item()
            val_loss /= max(len(val_loader), 1)

            print(f"  Época {epoch:3d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  [MOMENT] Early stopping en época {epoch}.")
                    break

        # Cargar mejor estado y evaluar métricas finales
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        preds_all, targets_all = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds_all.append(model(xb).cpu().numpy())
                targets_all.append(yb.numpy())

        p = np.vstack(preds_all)
        t = np.vstack(targets_all)
        n_params = model.count_parameters()

        fold_results.append({
            "fold": fold_idx,
            "mae_fisica": float(mean_absolute_error(t[:, 0], p[:, 0])),
            "rmse_fisica": float(np.sqrt(mean_squared_error(t[:, 0], p[:, 0]))),
            "r2_fisica": float(r2_score(t[:, 0], p[:, 0])),
            "mae_mental": float(mean_absolute_error(t[:, 1], p[:, 1])),
            "rmse_mental": float(np.sqrt(mean_squared_error(t[:, 1], p[:, 1]))),
            "r2_mental": float(r2_score(t[:, 1], p[:, 1])),
            "best_val_mse": best_val_loss,
            "num_params_total": n_params["total"],
            "num_params_trainable": n_params["trainable"],
            "n_val": len(val_idx),
        })

        print(
            f"  [MOMENT] Fold {fold_idx} — "
            f"MAE física: {fold_results[-1]['mae_fisica']:.4f} | "
            f"R² física: {fold_results[-1]['r2_fisica']:.4f} | "
            f"MAE mental: {fold_results[-1]['mae_mental']:.4f} | "
            f"R² mental: {fold_results[-1]['r2_mental']:.4f}"
        )

    total_time = time.time() - t_start
    return fold_results, total_time


# ---------------------------------------------------------------------------
# 5. Evaluador Zero-Shot de Google TimesFM 2.5 (PyTorch)
# ---------------------------------------------------------------------------

class TimesFMZeroShotEvaluator:
    """
    Evaluador zero-shot basado en Google TimesFM 2.5 (PyTorch) para predicción.

    Utiliza el paradigma de independencia de canales (channel independence) para extraer
    predicciones puntuales y de cuantiles para los 23 canales de FatigueSet.
    """

    def __init__(
        self,
        checkpoint: str = "google/timesfm-2.5-200m-pytorch",
        prediction_length: int = 1,
        max_context: int = 128,
        per_core_batch_size: int = 32,
        device: str = "cpu",
    ):
        self.checkpoint = checkpoint
        self.prediction_length = prediction_length
        self.max_context = max_context
        self.per_core_batch_size = per_core_batch_size
        self.device = device
        self._model = None

    def _load_model(self) -> None:
        """Carga el modelo TimesFM de forma perezosa."""
        if self._model is not None:
            return

        try:
            from timesfm import TimesFM_2p5_200M_torch, ForecastConfig
        except ImportError as e:
            raise ImportError(
                "El paquete 'timesfm' no está instalado. "
                "Instálalo con: pip install timesfm"
            ) from e

        print(f"[TimesFM] Cargando modelo desde '{self.checkpoint}'...")
        t0 = time.time()
        self._model = TimesFM_2p5_200M_torch.from_pretrained(
            self.checkpoint,
            torch_compile=False,
        )
        if hasattr(self._model, "to"):
            self._model = self._model.to(self.device)
            
        # Compilación estática/inicialización
        config = ForecastConfig(
            max_context=self.max_context,
            max_horizon=self.prediction_length,
            per_core_batch_size=self.per_core_batch_size,
        )
        self._model.compile(config)
        print(f"[TimesFM] Modelo y config listos en {time.time() - t0:.1f}s")

    def extract_features(self, X: np.ndarray, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrae características de predicción puntual y de cuantiles por lote (batch).

        Parámetros
        ----------
        X : np.ndarray, forma (N, seq_len, n_channels)
            Array de secuencias fisiológicas.
        batch_size : int
            Número de muestras procesadas simultáneamente.

        Retorna
        -------
        Tuple[np.ndarray, np.ndarray]
            - point_preds: forma (N, n_channels)
            - quantile_preds: forma (N, n_channels, 10)
        """
        self._load_model()

        N, seq_len, n_channels = X.shape
        point_preds = np.zeros((N, n_channels), dtype=np.float32)
        quantile_preds = np.zeros((N, n_channels, 10), dtype=np.float32)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = X[start:end]  # (B, seq_len, n_channels)
            B = end - start

            # Aplanar canales e inputs para procesamiento en batch
            flat_inputs = []
            for i in range(B):
                for ch in range(n_channels):
                    flat_inputs.append(batch[i, :, ch])

            # Inferencia
            point_pred, quant_pred = self._model.forecast(
                horizon=self.prediction_length,
                inputs=flat_inputs
            )

            # Des-aplanar
            for i in range(B):
                for ch in range(n_channels):
                    flat_idx = i * n_channels + ch
                    point_preds[start + i, ch] = point_pred[flat_idx, 0]
                    quantile_preds[start + i, ch, :] = quant_pred[flat_idx, 0, :]

        return point_preds, quantile_preds

    def fit_probe(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        """
        Entrena una regresión lineal de sonda (Ridge) sobre las características de TimesFM.

        Retorna
        -------
        Tuple[np.ndarray, np.ndarray, Ridge]
            Predicciones, características de validación y sonda entrenada.
        """
        from sklearn.linear_model import Ridge

        print("[TimesFM] Extrayendo características del conjunto de entrenamiento...")
        feats_train, _ = self.extract_features(X_train)
        print("[TimesFM] Extrayendo características del conjunto de validación...")
        feats_val, _ = self.extract_features(X_val)

        probe = Ridge(alpha=1.0)
        probe.fit(feats_train, y_train)
        y_pred = probe.predict(feats_val)

        return y_pred, feats_val, probe

