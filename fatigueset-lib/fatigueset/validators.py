"""
Validación de integridad del dataset FatigueSet.
Basado en la Celda 14 de FatigueSet Procesado de datos.ipynb
"""
from typing import List, Optional, Tuple

import pandas as pd

from .loader import FatigueSetLoader, PARTICIPANTES, SESIONES

ARCHIVOS_CRITICOS = [
    ('chest_physiology_summary.csv', 'Chest Physiology'),
    ('chest_raw_ecg.csv',            'ECG Raw'),
    ('wrist_hr.csv',                 'Wrist HR'),
    ('wrist_eda.csv',                'Wrist EDA'),
    ('wrist_skin_temperature.csv',   'Wrist Temp'),
    ('forehead_eeg_alpha_abs.csv',   'EEG Alpha'),
    ('forehead_eeg_beta_abs.csv',    'EEG Beta'),
    ('ear_ppg_left.csv',             'Ear PPG Left'),
    ('exp_fatigue.csv',              'Fatigue Scores'),
    ('exp_nback.csv',                'N-Back Task'),
    ('exp_crt.csv',                  'CRT Task'),
]


class DataValidator:
    """
    Validador genérico de DataFrames.
    """

    def validate_schema(self, df: pd.DataFrame, columns: list) -> bool:
        """
        Comprueba que el DataFrame contiene todas las columnas indicadas.

        Returns
        -------
        True si todas las columnas están presentes, False en caso contrario.
        """
        return all(col in df.columns for col in columns)

    def check_missing_values(self, df: pd.DataFrame) -> bool:
        """
        Detecta si el DataFrame contiene algún valor NaN.

        Returns
        -------
        True si hay valores ausentes, False si el DataFrame está completo.
        """
        return bool(df.isnull().any().any())

    def validate_data_quality(self, df: pd.DataFrame) -> bool:
        """
        Valida que cada columna tenga un tipo de dato coherente:
        una columna con valores numéricos no debe contener strings
        y viceversa (sin mezclar tipos).

        Returns
        -------
        True si todas las columnas son homogéneas, False si hay columnas mixtas.
        """
        for col in df.columns:
            if df[col].dtype == object:
                numeric = pd.to_numeric(df[col], errors='coerce')
                has_valid_numeric = numeric.notna().any()
                has_invalid = numeric.isna().any()
                if has_valid_numeric and has_invalid:
                    # Mezcla de números y cadenas → calidad baja
                    return False
        return True


class FatigueSetValidator:
    """
    Valida integridad de los datos del dataset FatigueSet.

    Parameters
    ----------
    loader : FatigueSetLoader
    umbral_nulos : float
        Porcentaje máximo de valores nulos permitido (default 5%).
    """

    def __init__(
        self,
        loader: FatigueSetLoader,
        umbral_nulos: float = 5.0,
    ):
        self.loader       = loader
        self.umbral_nulos = umbral_nulos

    def validar_archivo(
        self,
        participante: str,
        sesion: str,
        archivo: str,
        nombre: str,
    ) -> Optional[dict]:
        """Valida un único archivo para un participante y sesión."""
        df = self.loader.cargar_csv(participante, sesion, archivo, add_meta=False)
        if df is None:
            return None

        n_filas  = len(df)
        pct_nulos = (
            df.isnull().sum().sum() / (n_filas * len(df.columns)) * 100
            if n_filas > 0 else 0.0
        )
        n_dupl = df.duplicated().sum()
        ok     = pct_nulos < self.umbral_nulos and n_dupl == 0

        return {
            'Archivo':      nombre,
            'Participante': participante,
            'Sesión':       sesion,
            'Filas':        n_filas,
            'Nulos (%)':    round(pct_nulos, 2),
            'Duplicados':   int(n_dupl),
            'Estado':       '✓' if ok else '⚠️',
        }

    def validar_todos(
        self,
        archivos: Optional[List[Tuple[str, str]]] = None,
        participantes: Optional[List[str]] = None,
        sesiones: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Valida todos los archivos críticos para todos los
        participantes y sesiones.

        Replica la Celda 14 del notebook.

        Returns
        -------
        DataFrame con una fila por (archivo, participante, sesión).
        """
        archivos      = archivos      or ARCHIVOS_CRITICOS
        participantes = participantes or self.loader.participantes
        sesiones      = sesiones      or self.loader.sesiones

        filas = []
        for archivo, nombre in archivos:
            for pid in participantes:
                for ses in sesiones:
                    resultado = self.validar_archivo(pid, ses, archivo, nombre)
                    if resultado:
                        filas.append(resultado)

        return pd.DataFrame(filas)

    def resumen(
        self,
        df_validacion: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Genera resumen agregado y lista de problemas.

        Returns
        -------
        (resumen_por_archivo, df_problemas)
        """
        if df_validacion is None:
            df_validacion = self.validar_todos()

        resumen_df = df_validacion.groupby('Archivo').agg(
            Sesiones_OK    =('Estado', lambda x: (x == '✓').sum()),
            Sesiones_total =('Estado', 'count'),
            Nulos_pct_max  =('Nulos (%)', 'max'),
            Duplicados_max =('Duplicados', 'max'),
        ).round(2)

        problemas = df_validacion[df_validacion['Estado'] == '⚠️'].copy()

        return resumen_df, problemas

    def imprimir_resumen(self) -> None:
        """Imprime el resumen de validación en formato legible."""
        df_val            = self.validar_todos()
        resumen, problemas = self.resumen(df_val)

        print("=" * 60)
        print("VALIDACIÓN DE INTEGRIDAD DE DATOS")
        print("=" * 60)
        print(f"\n📊 Resumen por tipo de archivo (umbral nulos: {self.umbral_nulos}%):")
        print(resumen.to_string())

        if not problemas.empty:
            print(f"\n⚠️  Sesiones con problemas: {len(problemas)}")
            print(problemas.to_string(index=False))
        else:
            print("\n✓ Todos los archivos superan el umbral de calidad.")