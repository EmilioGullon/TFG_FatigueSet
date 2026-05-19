"""
Extracción de features para ML a partir de señales FatigueSet.
Complementa FatigueSetProcessor con estadísticas avanzadas por ventana temporal.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .loader import CANALES_EEG


class FeatureExtractor:
    """
    Extrae features estadísticas de señales fisiológicas.

    Parameters
    ----------
    percentiles : list
        Percentiles a calcular (default [25, 50, 75]).
    """

    def __init__(self, percentiles: Optional[List[int]] = None):
        self.percentiles = percentiles or [25, 50, 75]

    def extract_features(self, data: dict) -> dict:
        """
        Extrae la media de cada clave del diccionario de entrada.

        Parameters
        ----------
        data : dict
            {nombre: lista_valores}. Si algún valor es None lanza ValueError.

        Returns
        -------
        dict  {nombre_mean: valor, ...}

        Raises
        ------
        ValueError
            Si alguna lista contiene valores None.
        """
        for key, values in data.items():
            if any(v is None for v in values):
                raise ValueError("Invalid data: contains None values")

        return {f'{k}_mean': float(np.mean(v)) for k, v in data.items()}

    def compute_statistics(self, data: dict) -> dict:
        """
        Calcula estadísticas básicas (mean, std, min, max) para cada
        clave del diccionario.

        Parameters
        ----------
        data : dict
            {nombre: lista_valores}

        Returns
        -------
        dict  {nombre: {'mean': ..., 'std': ..., 'min': ..., 'max': ...}}
        """
        result = {}
        for key, values in data.items():
            arr = np.array(values, dtype=float)
            result[key] = {
                'mean': float(np.mean(arr)),
                'std':  float(np.std(arr, ddof=0)),
                'min':  float(np.min(arr)),
                'max':  float(np.max(arr)),
            }
        return result

    def features_serie(
        self,
        serie: pd.Series,
        prefijo: str = '',
    ) -> Dict[str, float]:
        """
        Calcula estadísticas descriptivas de una serie temporal.

        Returns
        -------
        dict con: media, std, min, max, rango, cv, p25, p50, p75
        """
        s = pd.to_numeric(serie, errors='coerce').dropna() # Convertir a numérico y eliminar NaN
        if len(s) == 0:
            return {} 

        feats = {
            f'{prefijo}media':  s.mean(),
            f'{prefijo}std':    s.std(),
            f'{prefijo}min':    s.min(),
            f'{prefijo}max':    s.max(),
            f'{prefijo}rango':  s.max() - s.min(),
            f'{prefijo}cv':     s.std() / s.mean() if s.mean() != 0 else np.nan, # Coeficiente de variación (std/mean)
        }
        for p in self.percentiles:
            feats[f'{prefijo}p{p}'] = s.quantile(p / 100) # Percentiles (p25, p50, p75)

        return feats
    # ------------------------------------------------------------------------
    # Features específicas para cada tipo de señal (EEG, HRV, EDA, cognitivas)
    # ------------------------------------------------------------------------
    def features_eeg_banda(
        self,
        df: pd.DataFrame,
        banda: str,
        canales: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Features EEG para una banda: media por canal + promedio global.
        """
        canales = canales or CANALES_EEG
        feats   = {}

        # Media por canal dentro de la banda
        for canal in canales:
            if canal in df.columns:
                feats.update(self.features_serie(
                    df[canal], prefijo=f'eeg_{banda}_{canal}_'))

        # Promedio global de la banda (media de los canales)
        cols_ok = [c for c in canales if c in df.columns]
        # Solo calculamos el global si hay al menos 2 canales disponibles
        if cols_ok:
            media_global = df[cols_ok].mean(axis=1)
            feats.update(self.features_serie(
                media_global, prefijo=f'eeg_{banda}_global_'))

        return feats

    def features_hrv(self, df_rr: pd.DataFrame, col_rr: str = 'rrInterval') -> Dict[str, float]:
        """
        Features básicas de HRV a partir de intervalos RR.

        Métricas: RMSSD, SDNN, pNN50
        """
        if col_rr not in df_rr.columns:
            return {}

        # Convertir a numérico, eliminar NaN y convertir a array
        rr = pd.to_numeric(df_rr[col_rr], errors='coerce').dropna().values
        if len(rr) < 2:
            return {}

        # Cálculo de métricas HRV
        diffs = np.diff(rr)
        return {
            'hrv_rmssd':  np.sqrt(np.mean(diffs ** 2)), # Raíz cuadrada de la media de las diferencias al cuadrado (RMSSD)
            'hrv_sdnn':   np.std(rr), # Desviación estándar de los intervalos RR (SDNN)
            'hrv_pnn50':  np.sum(np.abs(diffs) > 50) / len(diffs) * 100, # Porcentaje de diferencias mayores a 50 ms (pNN50)
            'hrv_mean_rr': np.mean(rr), # Media de los intervalos RR
        }

    def features_eda(
        self,
        df_eda: pd.DataFrame,
        col_eda: str = 'eda',
    ) -> Dict[str, float]:
        """
        Features de EDA (Electrodermal Activity):
        media, std, pendiente (tendencia), número de picos SCR.
        """
        if col_eda not in df_eda.columns:
            return {}

        # Convertir a numérico, eliminar NaN y convertir a serie
        s = pd.to_numeric(df_eda[col_eda], errors='coerce').dropna()
        if len(s) < 2:
            return {}

        # Estadísticas básicas de la serie EDA  (media, std, min, max, rango, cv, percentiles)
        feats = self.features_serie(s, prefijo='eda_')

        # Pendiente lineal (tendencia)
        x = np.arange(len(s))
        coef = np.polyfit(x, s.values, 1)
        feats['eda_pendiente'] = coef[0]

        # Número de picos SCR (cruces por encima de media + 1 std)
        umbral    = s.mean() + s.std()
        en_pico   = False
        n_picos   = 0
        for v in s:
            if v > umbral and not en_pico:
                n_picos += 1
                en_pico  = True
            elif v <= umbral:
                en_pico  = False
        feats['eda_n_picos_scr'] = n_picos

        return feats

    def features_cognitivas(
        self,
        df_nback: Optional[pd.DataFrame] = None,
        df_crt: Optional[pd.DataFrame]   = None,
    ) -> Dict[str, float]:
        """
        Features cognitivas combinadas de N-Back y CRT.
        Para N-Back: precisión (accuracy), número de errores, RT media y std.
        Para CRT: RT media y std.
        """
        feats = {}

        # N-Back: precisión (accuracy), número de errores, RT media y std
        if df_nback is not None and not df_nback.empty:
            if 'isCorrectResponse' in df_nback.columns:
                # Convertir a numérico (1 para correcto, 0 para incorrecto), ignorando errores de conversión
                acc = pd.to_numeric(
                    df_nback['isCorrectResponse'], errors='coerce')
                feats['nback_accuracy']  = acc.mean() * 100
                feats['nback_n_errores'] = (acc == 0).sum()

            rt_col = next(
                # Buscar columna de tiempo de respuesta (que contenga 'time' y 'response' en el nombre)
                (c for c in df_nback.columns
                 if 'time' in c.lower() and 'response' in c.lower()
                 and pd.to_numeric(df_nback[c], errors='coerce').notna().any()),
                None,
            )
            # Si encontramos una columna de RT válida, calculamos media y std de RT
            if rt_col: 
                rt = pd.to_numeric(df_nback[rt_col], errors='coerce')
                feats['nback_rt_media'] = rt.mean()
                feats['nback_rt_std']   = rt.std()

        # CRT: RT media y std (si hay columna de tiempo de respuesta válida)
        if df_crt is not None and not df_crt.empty:
            rt_col = next(
                (c for c in df_crt.columns
                 if 'time' in c.lower() and 'response' in c.lower()
                 and pd.to_numeric(df_crt[c], errors='coerce').notna().any()),
                None,
            )
            if rt_col:
                rt = pd.to_numeric(df_crt[rt_col], errors='coerce')
                feats['crt_rt_media'] = rt.mean()
                feats['crt_rt_std']   = rt.std()

        return feats