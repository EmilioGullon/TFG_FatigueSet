"""
Procesado de señales fisiológicas y tareas cognitivas de FatigueSet.
Basado en las celdas 7–15 de FatigueSet Procesado de datos.ipynb
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .loader import (
    CANALES_EEG,
    FASES_MEDICION,
    INTENSIDAD_NUM,
    PARTICIPANTES,
    SESIONES,
)


class DataProcessor:
    """
    Clase genérica de procesado de datos.
    """

    def process_data(self, data: dict) -> pd.DataFrame:
        """
        Crea un DataFrame a partir de un diccionario y añade una columna
        'processed_column' con la suma de todas las columnas numéricas por fila.

        Parameters
        ----------
        data : dict
            Diccionario {nombre_columna: lista_valores}.

        Returns
        -------
        pd.DataFrame con una columna adicional 'processed_column'.
        """
        df = pd.DataFrame(data)
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        df['processed_column'] = df[numeric_cols].sum(axis=1) if numeric_cols else 0
        return df

    def filter_data(self, data: dict, threshold: float = 0) -> pd.DataFrame:
        """
        Filtra el DataFrame resultante conservando solo las filas cuya primera
        columna es >= threshold.

        Returns
        -------
        pd.DataFrame filtrado.
        """
        df = pd.DataFrame(data)
        if df.empty:
            return df
        first_col = df.columns[0]
        return df[df[first_col] >= threshold].reset_index(drop=True)

    def aggregate_data(self, data: dict) -> pd.DataFrame:
        """
        Convierte el diccionario en un DataFrame sin modificaciones adicionales.
        Útil para calcular estadísticas globales sobre él.

        Returns
        -------
        pd.DataFrame.
        """
        return pd.DataFrame(data)


class FatigueSetProcessor:
    """
    Procesa los datos cargados por FatigueSetLoader:
    - Calcula fatigabilidad (deltas entre fases)
    - Construye el dataset ML agregado (108 filas)
    - Extrae métricas de tareas cognitivas
    """

    def __init__(self, intensidad_map: Dict[str, Dict[str, str]]):
        """
        Parameters
        ----------
        intensidad_map : dict
            Salida de FatigueSetLoader.construir_intensidad_map()
        """
        self.intensidad_map = intensidad_map

    # ------------------------------------------------------------------
    # Fatigabilidad
    # ------------------------------------------------------------------

    def calcular_fatigabilidad(
        self,
        df_fatiga: pd.DataFrame,
        participantes: Optional[List[str]] = None,
        sesiones: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Calcula los deltas de fatiga entre fases M1/M2/M3
        para cada (participante, sesión).

        Replica la Celda 7 del notebook.

        Returns
        -------
        DataFrame con 36 filas (12 participantes × 3 sesiones).
        Columnas:
            delta_fisica_ejercicio  : M2-M1 física
            delta_mental_ejercicio  : M2-M1 mental
            delta_fisica_cognitivo  : M3-M2 física
            delta_mental_cognitivo  : M3-M2 mental
            delta_fisica_total      : M3-M1 física
            delta_mental_total      : M3-M1 mental
            fisica_M1/M2/M3, mental_M1/M2/M3  (scores absolutos)
        """
        pids = participantes or PARTICIPANTES
        sess = sesiones      or SESIONES
        resultados = []

        for pid in pids:
            for ses in sess:
                df_s = df_fatiga[
                    (df_fatiga['participante'] == pid) &
                    (df_fatiga['sesion'] == ses)
                ].sort_values('measurementNumber')

                m = {row['measurementNumber']: row
                     for _, row in df_s.iterrows()}

                if not (0 in m and 1 in m and 2 in m):
                    continue

                intensidad = self.intensidad_map.get(pid, {}).get(ses, 'unknown')
                resultados.append({
                    'participante':   pid,
                    'sesion':         ses,
                    'intensidad':     intensidad,
                    'intensidad_num': INTENSIDAD_NUM.get(intensidad, 0),
                    # Efectos por fase
                    'delta_fisica_ejercicio':  m[1]['physicalFatigueScore'] - m[0]['physicalFatigueScore'],
                    'delta_mental_ejercicio':  m[1]['mentalFatigueScore']   - m[0]['mentalFatigueScore'],
                    'delta_fisica_cognitivo':  m[2]['physicalFatigueScore'] - m[1]['physicalFatigueScore'],
                    'delta_mental_cognitivo':  m[2]['mentalFatigueScore']   - m[1]['mentalFatigueScore'],
                    'delta_fisica_total':      m[2]['physicalFatigueScore'] - m[0]['physicalFatigueScore'],
                    'delta_mental_total':      m[2]['mentalFatigueScore']   - m[0]['mentalFatigueScore'],
                    # Scores absolutos
                    'fisica_M1': m[0]['physicalFatigueScore'],
                    'fisica_M2': m[1]['physicalFatigueScore'],
                    'fisica_M3': m[2]['physicalFatigueScore'],
                    'mental_M1': m[0]['mentalFatigueScore'],
                    'mental_M2': m[1]['mentalFatigueScore'],
                    'mental_M3': m[2]['mentalFatigueScore'],
                })

        return pd.DataFrame(resultados)

    # ------------------------------------------------------------------
    # Segmentación temporal por fase
    # ------------------------------------------------------------------

    @staticmethod
    def segmentar_por_fase(
        df: pd.DataFrame,
        fase_num: int,
        n_fases: int = 3,
    ) -> pd.DataFrame:
        """
        Divide un DataFrame de serie temporal en tercios y devuelve
        el segmento correspondiente a fase_num (0, 1 o 2).

        Nota: aproximación temporal. Para máxima precisión usar
        los timestamps de exp_markers.csv.
        """
        n = len(df)
        if n == 0:
            return df
        s = int(fase_num * n / n_fases)
        e = int((fase_num + 1) * n / n_fases)
        return df.iloc[s:e]

    # ------------------------------------------------------------------
    # Métricas de señales fisiológicas
    # ------------------------------------------------------------------

    @staticmethod
    def metricas_segmento(
        df: pd.DataFrame,
        columnas: List[str],
        prefijo: str = '',
    ) -> Dict[str, float]:
        """
        Calcula media y std de las columnas indicadas sobre un segmento.

        Returns
        -------
        dict  {prefijo_col_media: float, prefijo_col_std: float}
        """
        metricas = {}
        for col in columnas:
            if col in df.columns:
                serie = pd.to_numeric(df[col], errors='coerce').dropna()
                metricas[f'{prefijo}{col}_media'] = serie.mean() if len(serie) > 0 else np.nan
                metricas[f'{prefijo}{col}_std']   = serie.std()  if len(serie) > 0 else np.nan
        return metricas

    @staticmethod
    def metricas_eeg_segmento(
        df: pd.DataFrame,
        banda: str,
        canales: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Media de todos los canales EEG para un segmento y banda."""
        canales = canales or CANALES_EEG
        cols_ok = [c for c in canales if c in df.columns]
        if not cols_ok:
            return {}
        return {f'eeg_{banda}_media': df[cols_ok].mean().mean()}

    # ------------------------------------------------------------------
    # Métricas de tareas cognitivas
    # ------------------------------------------------------------------

    @staticmethod
    def metricas_nback(df_nback_fase: pd.DataFrame) -> Dict[str, float]:
        """
        Extrae accuracy y RT del N-Back para un subset (participante,sesión,fase).
        Replica lógica de Celda 13 del notebook.
        """
        metricas = {}
        if df_nback_fase.empty:
            return metricas

        if 'isCorrectResponse' in df_nback_fase.columns:
            metricas['nback_accuracy'] = (
                pd.to_numeric(df_nback_fase['isCorrectResponse'], errors='coerce').mean() * 100
            )

        rt_col = next(
            (c for c in df_nback_fase.columns
             if 'time' in c.lower() and 'response' in c.lower()
             and pd.to_numeric(df_nback_fase[c], errors='coerce').notna().any()),
            None
        )
        if rt_col:
            metricas['nback_rt'] = pd.to_numeric(
                df_nback_fase[rt_col], errors='coerce').mean()

        return metricas

    @staticmethod
    def metricas_crt(df_crt_fase: pd.DataFrame) -> Dict[str, float]:
        """
        Extrae RT (y accuracy si disponible) del CRT para un subset.
        Replica lógica de Celda 13 del notebook.
        Evita el TypeError de columnas de texto tipo correctKey.
        """
        metricas = {}
        if df_crt_fase.empty:
            return metricas

        rt_col = next(
            (c for c in df_crt_fase.columns
             if 'time' in c.lower() and 'response' in c.lower()
             and pd.to_numeric(df_crt_fase[c], errors='coerce').notna().any()),
            None
        )
        if rt_col:
            metricas['crt_rt'] = pd.to_numeric(
                df_crt_fase[rt_col], errors='coerce').mean()

        # Solo columnas realmente numéricas/booleanas con 'correct'
        acc_col = next(
            (c for c in df_crt_fase.columns
             if 'correct' in c.lower()
             and df_crt_fase[c].dtype in [bool, np.bool_, np.int64, np.float64]
             and c.lower() not in ['correctkey', 'correctans', 'correctresponse']),
            None
        )
        if acc_col:
            metricas['crt_accuracy'] = (
                pd.to_numeric(df_crt_fase[acc_col], errors='coerce').mean() * 100
            )

        return metricas

    # ------------------------------------------------------------------
    # Dataset ML agregado (Celda 15)
    # ------------------------------------------------------------------

    def construir_dataset_ml(
        self,
        df_fatiga: Optional[pd.DataFrame]      = None,
        df_chest: Optional[pd.DataFrame]        = None,
        df_wrist_eda: Optional[pd.DataFrame]    = None,
        eeg_data: Optional[Dict]                = None,
        df_nback: Optional[pd.DataFrame]        = None,
        df_crt: Optional[pd.DataFrame]          = None,
        participantes: Optional[List[str]]      = None,
        sesiones: Optional[List[str]]           = None,
    ) -> pd.DataFrame:
        """
        Construye el dataset ML con granularidad:
            12 participantes × 3 sesiones × 3 fases = 108 filas.

        Replica la Celda 15 del notebook.

        Parameters
        ----------
        df_fatiga    : salida de loader.cargar_fatiga()
        df_chest     : salida de loader.cargar_chest()
        df_wrist_eda : salida de loader.cargar_wrist()['eda']
        eeg_data     : salida de loader.cargar_eeg()
        df_nback     : salida de loader.cargar_nback()
        df_crt       : salida de loader.cargar_crt()

        Returns
        -------
        DataFrame de 108 filas listo para ML.
        """
        pids  = participantes or PARTICIPANTES
        sess  = sesiones      or SESIONES
        datos = []

        for pid in pids:
            for ses in sess:
                intensidad     = self.intensidad_map.get(pid, {}).get(ses, 'unknown')
                intensidad_num = INTENSIDAD_NUM.get(intensidad, 0)

                for fase_num, fase_nombre in FASES_MEDICION.items():
                    fila: Dict = {
                        'participante':   pid,
                        'sesion':         ses,
                        'intensidad':     intensidad,
                        'intensidad_num': intensidad_num,
                        'fase':           fase_nombre,
                        'fase_num':       fase_num,
                    }

                    # --- Fatiga subjetiva (VAS) ---
                    if df_fatiga is not None:
                        df_f = df_fatiga[
                            (df_fatiga['participante'] == pid) &
                            (df_fatiga['sesion'] == ses) &
                            (df_fatiga['measurementNumber'] == fase_num)
                        ]
                        if not df_f.empty:
                            fila['fatiga_fisica'] = df_f['physicalFatigueScore'].iloc[0]
                            fila['fatiga_mental'] = df_f['mentalFatigueScore'].iloc[0]

                    # --- Chest / Zephyr ---
                    if df_chest is not None:
                        seg = self._get_segmento(df_chest, pid, ses, fase_num)
                        fila.update(self.metricas_segmento(
                            seg, ['hr', 'br', 'hrv']))

                    # --- EDA / Empatica ---
                    if df_wrist_eda is not None:
                        seg = self._get_segmento(df_wrist_eda, pid, ses, fase_num)
                        fila.update(self.metricas_segmento(seg, ['eda']))

                    # --- EEG / Muse ---
                    if eeg_data:
                        for banda in ['alpha', 'beta', 'theta']:
                            if banda in eeg_data and eeg_data[banda] is not None:
                                seg = self._get_segmento(
                                    eeg_data[banda], pid, ses, fase_num)
                                fila.update(self.metricas_eeg_segmento(seg, banda))

                    # --- N-Back ---
                    if df_nback is not None and 'fase' in df_nback.columns:
                        sub = df_nback[
                            (df_nback['participante'] == pid) &
                            (df_nback['sesion'] == ses) &
                            (df_nback['fase'] == fase_nombre)
                        ]
                        fila.update(self.metricas_nback(sub))

                    # --- CRT ---
                    if df_crt is not None and 'fase' in df_crt.columns:
                        sub = df_crt[
                            (df_crt['participante'] == pid) &
                            (df_crt['sesion'] == ses) &
                            (df_crt['fase'] == fase_nombre)
                        ]
                        fila.update(self.metricas_crt(sub))

                    datos.append(fila)

        return pd.DataFrame(datos)

    def _get_segmento(
        self,
        df: pd.DataFrame,
        pid: str,
        ses: str,
        fase_num: int,
    ) -> pd.DataFrame:
        """Filtra por participante/sesión y segmenta por tercio temporal."""
        sub = df[
            (df['participante'] == pid) &
            (df['sesion'] == ses)
        ].reset_index(drop=True)
        return self.segmentar_por_fase(sub, fase_num)