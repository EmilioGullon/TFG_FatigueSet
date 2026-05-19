"""
Pipeline de alto nivel para FatigueSet.

Orquesta carga, validación, fatigabilidad, dataset ML, normalización,
ventaneo y resumen de correlaciones usando la API pública de la librería.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .features import FeatureExtractor
from .loader import FatigueSetLoader, PARTICIPANTES, SESIONES
from .processor import FatigueSetProcessor
from .validators import FatigueSetValidator
from .utils import top_correlaciones


class FatigueSetPipeline:
    """
    Orquestador de extremo a extremo para el dataset FatigueSet.

    El pipeline cubre los pasos que aparecen repartidos en los notebooks
    del proyecto:
    - carga completa del dataset
    - validación de integridad
    - cálculo de fatigabilidad
    - construcción del dataset ML agregado
    - normalización
    - extracción de features por ventanas
    - correlaciones con fatiga física y mental
    """

    def __init__(
        self,
        dataset_path: str = 'fatigueset',
        participantes: Optional[List[str]] = None,
        sesiones: Optional[List[str]] = None,
        bandas_eeg: Optional[List[str]] = None,
        umbral_nulos: float = 5.0,
    ):
        self.dataset_path = dataset_path
        self.participantes = participantes or PARTICIPANTES
        self.sesiones = sesiones or SESIONES
        self.bandas_eeg = bandas_eeg

        self.loader = FatigueSetLoader(
            base_path=dataset_path,
            participantes=self.participantes,
            sesiones=self.sesiones,
        )
        self.validator = FatigueSetValidator(self.loader, umbral_nulos=umbral_nulos)
        self.feature_extractor = FeatureExtractor()
        self._processor: Optional[FatigueSetProcessor] = None

    @property
    def processor(self) -> FatigueSetProcessor:
        if self._processor is None:
            intensidad_map = self.loader.construir_intensidad_map()
            self._processor = FatigueSetProcessor(intensidad_map)
        return self._processor

    def cargar_dataset(self, verbose: bool = True) -> Dict:
        """Carga el dataset completo con la API de FatigueSetLoader."""
        if self.bandas_eeg is not None:
            self.loader.construir_intensidad_map()
        return self.loader.cargar_todo(verbose=verbose)

    def validar_dataset(
        self,
        archivos: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Devuelve validación detallada, resumen agregado y problemas detectados."""
        df_validacion = self.validator.validar_todos(
            archivos=archivos,
            participantes=self.participantes,
            sesiones=self.sesiones,
        )
        resumen, problemas = self.validator.resumen(df_validacion)
        return {
            'validacion': df_validacion,
            'resumen': resumen,
            'problemas': problemas,
        }

    def calcular_fatigabilidad(self, df_fatiga: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Calcula deltas de fatiga entre fases para cada participante y sesión."""
        if df_fatiga is None or df_fatiga.empty:
            return pd.DataFrame()
        return self.processor.calcular_fatigabilidad(
            df_fatiga,
            participantes=self.participantes,
            sesiones=self.sesiones,
        )

    def construir_dataset_ml(self, dataset: Dict) -> pd.DataFrame:
        """Construye el dataset agregado por participante, sesión y fase."""
        df_ml = self.processor.construir_dataset_ml(
            df_fatiga=dataset.get('fatiga'),
            df_chest=dataset.get('chest'),
            df_wrist_eda=(dataset.get('wrist') or {}).get('eda'),
            eeg_data=dataset.get('eeg'),
            df_nback=dataset.get('nback'),
            df_crt=dataset.get('crt'),
            participantes=self.participantes,
            sesiones=self.sesiones,
        )

        df_fatigabilidad = self.calcular_fatigabilidad(dataset.get('fatiga'))
        if not df_fatigabilidad.empty and not df_ml.empty:
            claves = ['participante', 'sesion', 'intensidad', 'intensidad_num']
            comunes = [col for col in claves if col in df_fatigabilidad.columns and col in df_ml.columns]
            if comunes:
                df_ml = df_ml.merge(df_fatigabilidad, on=comunes, how='left')

        return df_ml

    def normalizar_dataframe(
        self,
        df: pd.DataFrame,
        metodo: str = 'zscore',
        group_cols: Optional[Sequence[str]] = ('participante', 'sesion'),
        exclude_columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Normaliza columnas numéricas por grupo manteniendo las columnas de identidad."""
        if df is None or df.empty:
            return pd.DataFrame() if df is None else df.copy()

        out = df.copy()
        exclude = set(exclude_columns or {
            'participante', 'sesion', 'intensidad', 'intensidad_num',
            'fase', 'fase_num', 'fatiga_fisica', 'fatiga_mental',
        })

        numeric_cols = [
            col for col in out.select_dtypes(include=[np.number]).columns
            if col not in exclude
        ]
        if not numeric_cols:
            return out

        valid_group_cols = [col for col in (group_cols or []) if col in out.columns]
        if not valid_group_cols:
            return self._normalizar_columnas(out, numeric_cols, metodo)

        partes = []
        for _, sub in out.groupby(valid_group_cols, dropna=False, sort=False):
            partes.append(self._normalizar_columnas(sub.copy(), numeric_cols, metodo))
        return pd.concat(partes, ignore_index=True) if partes else out

    def crear_ventanas(
        self,
        df: pd.DataFrame,
        window_size: int = 64,
        step: int = 32,
        group_cols: Optional[Sequence[str]] = ('participante', 'sesion', 'fase'),
        feature_columns: Optional[Sequence[str]] = None,
        exclude_columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Convierte series en ventanas y resume cada una con estadísticos básicos."""
        if df is None or df.empty:
            return pd.DataFrame()

        out = []
        exclude = set(exclude_columns or {
            'participante', 'sesion', 'intensidad', 'intensidad_num',
            'fase', 'fase_num', 'fatiga_fisica', 'fatiga_mental',
        })

        valid_group_cols = [col for col in (group_cols or []) if col in df.columns]
        grouped = [((), df)] if not valid_group_cols else df.groupby(valid_group_cols, dropna=False, sort=False)

        for group_key, sub in grouped:
            sub = sub.reset_index(drop=True)
            if len(sub) < window_size:
                continue

            if not isinstance(group_key, tuple):
                group_key = (group_key,)

            columns = list(feature_columns or sub.select_dtypes(include=[np.number]).columns)
            columns = [col for col in columns if col not in exclude]

            for start in range(0, len(sub) - window_size + 1, step):
                window = sub.iloc[start:start + window_size]
                fila: Dict[str, object] = {
                    'ventana_inicio': start,
                    'ventana_fin': start + window_size,
                    'ventana_tamano': window_size,
                }

                for col_name, value in zip(valid_group_cols, group_key):
                    fila[col_name] = value

                for col in columns:
                    serie = pd.to_numeric(window[col], errors='coerce').dropna()
                    if len(serie) == 0:
                        continue
                    fila.update(self.feature_extractor.features_serie(serie, prefijo=f'{col}_'))

                out.append(fila)

        return pd.DataFrame(out)

    def resumen_correlaciones(
        self,
        df_ml: pd.DataFrame,
        top_n: int = 10,
    ) -> Dict[str, pd.Series]:
        """Devuelve las features más correlacionadas con fatiga física y mental."""
        if df_ml is None or df_ml.empty:
            return {}

        resultado: Dict[str, pd.Series] = {}
        for target in ['fatiga_mental', 'fatiga_fisica']:
            if target in df_ml.columns:
                resultado[target] = top_correlaciones(df_ml, target=target, n=top_n)
        return resultado

    def ejecutar(
        self,
        verbose: bool = True,
        incluir_ventanas: bool = False,
        window_size: int = 64,
        step: int = 32,
        normalizar: bool = True,
        metodo_normalizacion: str = 'zscore',
    ) -> Dict[str, object]:
        """Ejecuta el pipeline completo y devuelve todos los artefactos generados."""
        dataset = self.cargar_dataset(verbose=verbose)
        validacion = self.validar_dataset()
        df_fatigabilidad = self.calcular_fatigabilidad(dataset.get('fatiga'))
        df_ml = self.construir_dataset_ml(dataset)

        if not df_fatigabilidad.empty and not df_ml.empty:
            claves = ['participante', 'sesion', 'intensidad', 'intensidad_num']
            comunes = [col for col in claves if col in df_fatigabilidad.columns and col in df_ml.columns]
            if comunes and not set(df_fatigabilidad.columns).issubset(df_ml.columns):
                df_ml = df_ml.merge(df_fatigabilidad, on=comunes, how='left', suffixes=('', '_fatigabilidad'))

        df_ml_normalizado = self.normalizar_dataframe(df_ml, metodo=metodo_normalizacion) if normalizar else pd.DataFrame()
        ventanas = self.crear_ventanas(df_ml, window_size=window_size, step=step) if incluir_ventanas else pd.DataFrame()
        correlaciones = self.resumen_correlaciones(df_ml)

        return {
            'dataset': dataset,
            'validacion': validacion['validacion'],
            'resumen_validacion': validacion['resumen'],
            'problemas_validacion': validacion['problemas'],
            'fatigabilidad': df_fatigabilidad,
            'ml': df_ml,
            'ml_normalizado': df_ml_normalizado,
            'ventanas': ventanas,
            'correlaciones': correlaciones,
        }

    @staticmethod
    def _normalizar_columnas(df: pd.DataFrame, columnas: Sequence[str], metodo: str) -> pd.DataFrame:
        for col in columnas:
            serie = pd.to_numeric(df[col], errors='coerce')
            if metodo == 'minmax':
                minimo = serie.min()
                rango = serie.max() - minimo
                df[col] = 0.0 if pd.isna(rango) or rango == 0 else (serie - minimo) / rango
            else:
                media = serie.mean()
                std = serie.std(ddof=0)
                df[col] = 0.0 if pd.isna(std) or std == 0 else (serie - media) / std
        return df