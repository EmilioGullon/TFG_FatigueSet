"""
Módulo de carga de datos del dataset FatigueSet.
Basado en el pipeline de FatigueSet Procesado de datos.ipynb
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


PARTICIPANTES = [f'{i:02d}' for i in range(1, 13)]
SESIONES      = ['01', '02', '03']

FASES_MEDICION = {
    0: 'M1_baseline',
    1: 'M2_post_ejercicio',
    2: 'M3_post_fatiga_mental'
}

INTENSIDAD_NUM = {'low': 1, 'medium': 2, 'high': 3}

CANALES_EEG = ['TP9', 'AF7', 'AF8', 'TP10']

ARCHIVOS_POR_SENSOR = {
    'chest':       ['chest_physiology_summary.csv', 'chest_raw_ecg.csv',
                    'chest_raw_acc.csv', 'chest_raw_breathing.csv',
                    'chest_rr_interval.csv', 'chest_bb_interval.csv',
                    'chest_sensor_summary.csv'],
    'zephyr':      ['zephyr_activity_summary.csv', 'zephyr_device_status.csv'],
    'wrist':       ['wrist_hr.csv', 'wrist_eda.csv', 'wrist_bvp.csv',
                    'wrist_skin_temperature.csv', 'wrist_acc.csv', 'wrist_ibi.csv'],
    'forehead':    ['forehead_eeg_alpha_abs.csv', 'forehead_eeg_beta_abs.csv',
                    'forehead_eeg_delta_abs.csv', 'forehead_eeg_gamma_abs.csv',
                    'forehead_eeg_theta_abs.csv', 'forehead_eeg_raw.csv',
                    'forehead_acc.csv', 'forehead_gyro.csv'],
    'muse':        ['muse_blinks.csv', 'muse_jaw_clenches.csv',
                    'muse_device_battery.csv', 'muse_device_fit.csv',
                    'muse_device_touch.csv'],
    'ear':         ['ear_ppg_left.csv', 'ear_ppg_right.csv',
                    'ear_acc_left.csv', 'ear_acc_right.csv',
                    'ear_gyro_left.csv', 'ear_gyro_right.csv'],
    'experimental':['exp_fatigue.csv', 'exp_markers.csv',
                    'exp_nback.csv', 'exp_crt.csv', 'exp_task_switch.csv'],
}


class DataLoader:
    """
    Clase genérica para la carga de archivos CSV.

    Parameters
    ----------
    base_path : str | Path, optional
        Directorio base desde el que resolver rutas relativas.
        Por defecto el directorio de trabajo actual.
    """

    def __init__(self, base_path: str | Path | None = None):
        # Por defecto, resolver rutas relativas desde la raíz del paquete
        # fatigueset-lib (útil para los tests que pasan 'data/sample/...').
        if base_path is None:
            self.base_path = Path(__file__).resolve().parent.parent
        else:
            self.base_path = Path(base_path)

    def load_csv(self, path: str | Path) -> pd.DataFrame:
        """
        Lee un archivo CSV y devuelve un DataFrame.

        Raises
        ------
        ValueError
            Si el archivo no tiene extensión .csv.
        FileNotFoundError
            Si el archivo no existe.
        """
        p = Path(path)
        if not p.is_absolute():
            # Si la ruta ya existe tal cual (relativa al CWD), úsala directamente.
            # Si no, resuélvela desde base_path.
            if not p.exists():
                p = self.base_path / p

        if p.suffix.lower() != '.csv':
            raise ValueError(f"Solo se admiten archivos .csv: {path}")
        if not p.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {p}")

        return pd.read_csv(p)

    def load_all(self, dir_path: str | Path) -> list:
        """
        Lee todos los archivos .csv de un directorio y los devuelve
        como lista de DataFrames.

        Returns
        -------
        list[pd.DataFrame]
        """
        d = Path(dir_path)
        if not d.is_absolute():
            if not d.exists():
                d = self.base_path / d

        csv_files = list(d.glob('*.csv'))
        return [pd.read_csv(f) for f in sorted(csv_files)]


class FatigueSetLoader:
    """
    Carga datos del dataset FatigueSet para todos los participantes y sesiones.

    Parameters
    ----------
    base_path : str | Path
        Ruta a la carpeta raíz del dataset (que contiene 01/, 02/, ..., 12/).
    participantes : list, optional
        Lista de IDs de participantes. Por defecto todos ('01'...'12').
    sesiones : list, optional
        Lista de sesiones. Por defecto ['01','02','03'].
    """

    def __init__(
        self,
        base_path: str | Path = 'fatigueset',
        participantes: Optional[List[str]] = None,
        sesiones: Optional[List[str]] = None,
    ):
        self.base_path    = Path(base_path)
        self.participantes = participantes or PARTICIPANTES
        self.sesiones      = sesiones or SESIONES
        self._intensidad_map: Dict[str, Dict[str, str]] = {}

        if not self.base_path.exists():
            raise FileNotFoundError(f"No se encontró el dataset en: {self.base_path}")

    # ------------------------------------------------------------------
    # Metadata e intensidades
    # ------------------------------------------------------------------

    def cargar_metadata(self) -> pd.DataFrame:
        """Carga fatigueset/metadata.csv y devuelve el DataFrame."""
        path = self.base_path / 'metadata.csv'
        if not path.exists():
            raise FileNotFoundError(f"metadata.csv no encontrado en {self.base_path}")
        return pd.read_csv(path)

    def construir_intensidad_map(self) -> Dict[str, Dict[str, str]]:
        """
        Construye el mapeo participante × sesión → intensidad
        a partir de metadata.csv.

        metadata.csv tiene columnas:
            participant_id, low_session, medium_session, high_session

        Returns
        -------
        dict  {pid: {sesion: intensidad}}
        """
        df_meta = self.cargar_metadata()
        mapa = {}

        for _, row in df_meta.iterrows():
            pid = f"{int(row['participant_id']):02d}"
            mapa[pid] = {
                f"{int(row['low_session']):02d}":    'low',
                f"{int(row['medium_session']):02d}": 'medium',
                f"{int(row['high_session']):02d}":   'high',
            }

        # Validar
        for pid, m in mapa.items():
            assert set(m.values()) == {'low', 'medium', 'high'}, \
                f"P{pid}: mapeo de intensidades incompleto → {m}"

        self._intensidad_map = mapa
        return mapa

    def get_intensidad(self, participante: str, sesion: str) -> str:
        """Devuelve la intensidad ('low'|'medium'|'high') de una sesión."""
        if not self._intensidad_map:
            self.construir_intensidad_map()
        return self._intensidad_map.get(participante, {}).get(sesion, 'unknown')

    # ------------------------------------------------------------------
    # Carga de un único archivo
    # ------------------------------------------------------------------

    def cargar_csv(
        self,
        participante: str,
        sesion: str,
        archivo: str,
        add_meta: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Carga fatigueset/<participante>/<sesion>/<archivo>.

        Parameters
        ----------
        participante : str   p.ej. '01'
        sesion       : str   p.ej. '02'
        archivo      : str   p.ej. 'wrist_hr.csv'
        add_meta     : bool  si True añade columnas participante/sesion/intensidad

        Returns
        -------
        DataFrame o None si el archivo no existe.
        """
        ruta = self.base_path / participante / sesion / archivo
        if not ruta.exists():
            return None

        df = pd.read_csv(ruta)

        if add_meta:
            intensidad = self.get_intensidad(participante, sesion)
            df['participante']   = participante
            df['sesion']         = sesion
            df['intensidad']     = intensidad
            df['intensidad_num'] = INTENSIDAD_NUM.get(intensidad, 0)

        return df

    # ------------------------------------------------------------------
    # Carga de todos los participantes / sesiones
    # ------------------------------------------------------------------

    def cargar_todos(
        self,
        archivo: str,
        verbose: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Carga un archivo de todos los participantes y sesiones configurados.

        Returns
        -------
        DataFrame concatenado o None si no existe ninguno.
        """
        dfs, faltantes = [], []

        for pid in self.participantes:
            for ses in self.sesiones:
                df = self.cargar_csv(pid, ses, archivo)
                if df is not None:
                    dfs.append(df)
                else:
                    faltantes.append(f"P{pid}/S{ses}")

        if verbose and faltantes:
            preview = faltantes[:5]
            extra   = f"... (+{len(faltantes)-5})" if len(faltantes) > 5 else ""
            print(f"  ⚠️  {archivo}: no encontrado en {preview}{extra}")

        return pd.concat(dfs, ignore_index=True) if dfs else None

    # ------------------------------------------------------------------
    # Helpers de señales específicas
    # ------------------------------------------------------------------

    def cargar_fatiga(self, verbose: bool = True) -> Optional[pd.DataFrame]:
        """Carga exp_fatigue.csv y añade columna 'fase'."""
        df = self.cargar_todos('exp_fatigue.csv', verbose=verbose)
        if df is not None and 'measurementNumber' in df.columns:
            df['fase'] = df['measurementNumber'].map(FASES_MEDICION)
        return df

    def cargar_markers(self, verbose: bool = True) -> Optional[pd.DataFrame]:
        """Carga exp_markers.csv y convierte utcTime a datetime."""
        df = self.cargar_todos('exp_markers.csv', verbose=verbose)
        if df is not None and 'utcTime' in df.columns:
            df['datetime'] = pd.to_datetime(df['utcTime'], unit='ms')
        return df

    def cargar_chest(self, verbose: bool = True) -> Optional[pd.DataFrame]:
        """Carga chest_physiology_summary.csv con timestamp convertido."""
        df = self.cargar_todos('chest_physiology_summary.csv', verbose=verbose)
        return convertir_timestamp(df)

    def cargar_wrist(self, verbose: bool = True) -> Dict[str, Optional[pd.DataFrame]]:
        """Carga todos los sensores de muñeca (Empatica E4)."""
        return {
            'hr':   self.cargar_todos('wrist_hr.csv', verbose=verbose),
            'eda':  self.cargar_todos('wrist_eda.csv', verbose=verbose),
            'bvp':  self.cargar_todos('wrist_bvp.csv', verbose=verbose),
            'temp': self.cargar_todos('wrist_skin_temperature.csv', verbose=verbose),
            'acc':  self.cargar_todos('wrist_acc.csv', verbose=verbose),
            'ibi':  self.cargar_todos('wrist_ibi.csv', verbose=verbose),
        }

    def cargar_eeg(
        self,
        bandas: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Carga bandas EEG del Muse.

        Parameters
        ----------
        bandas : list, optional
            Subconjunto de ['alpha','beta','delta','gamma','theta'].
            Por defecto todas.
        """
        bandas = bandas or ['alpha', 'beta', 'delta', 'gamma', 'theta']
        resultado = {}
        for banda in bandas:
            df = self.cargar_todos(f'forehead_eeg_{banda}_abs.csv', verbose=verbose)
            if df is not None:
                canales_ok = [c for c in CANALES_EEG if c in df.columns]
                if canales_ok:
                    df['promedio_canales'] = df[canales_ok].mean(axis=1)
            resultado[banda] = df
        return resultado

    def cargar_ear_ppg(self, verbose: bool = True) -> Dict[str, Optional[pd.DataFrame]]:
        """Carga PPG auricular izquierdo y derecho (Nokia eSense)."""
        return {
            'left':  self.cargar_todos('ear_ppg_left.csv', verbose=verbose),
            'right': self.cargar_todos('ear_ppg_right.csv', verbose=verbose),
        }

    def cargar_nback(self, verbose: bool = True) -> Optional[pd.DataFrame]:
        """Carga exp_nback.csv y añade columna 'fase'."""
        df = self.cargar_todos('exp_nback.csv', verbose=verbose)
        if df is not None and 'measurementNumber' in df.columns:
            df['fase'] = df['measurementNumber'].map(FASES_MEDICION)
        return df

    def cargar_crt(self, verbose: bool = True) -> Optional[pd.DataFrame]:
        """Carga exp_crt.csv y añade columna 'fase'."""
        df = self.cargar_todos('exp_crt.csv', verbose=verbose)
        if df is not None and 'measurementNumber' in df.columns:
            df['fase'] = df['measurementNumber'].map(FASES_MEDICION)
        return df

    def cargar_task_switch(self, verbose: bool = True) -> Optional[pd.DataFrame]:
        """Carga exp_task_switch.csv (tarea de inducción de fatiga mental S3)."""
        return self.cargar_todos('exp_task_switch.csv', verbose=verbose)

    # ------------------------------------------------------------------
    # Carga completa del dataset
    # ------------------------------------------------------------------

    def cargar_todo(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Carga todas las señales disponibles de una vez.

        Returns
        -------
        dict con claves:
            metadata, fatiga, markers,
            chest, wrist, eeg, ear,
            nback, crt, task_switch
        """
        if verbose:
            print("Cargando dataset FatigueSet completo...")

        if not self._intensidad_map:
            self.construir_intensidad_map()

        dataset = {
            'metadata':    self.cargar_metadata(),
            'fatiga':      self.cargar_fatiga(verbose=verbose),
            'markers':     self.cargar_markers(verbose=verbose),
            'chest':       self.cargar_chest(verbose=verbose),
            'wrist':       self.cargar_wrist(verbose=verbose),
            'eeg':         self.cargar_eeg(verbose=verbose),
            'ear':         self.cargar_ear_ppg(verbose=verbose),
            'nback':       self.cargar_nback(verbose=verbose),
            'crt':         self.cargar_crt(verbose=verbose),
            'task_switch': self.cargar_task_switch(verbose=verbose),
        }

        if verbose:
            self._print_resumen(dataset)

        return dataset

    def _print_resumen(self, dataset: dict):
        print("\n📋 Dataset cargado:")
        print(f"   Participantes: {len(self.participantes)}")
        print(f"   Sesiones:      {len(self.participantes) * len(self.sesiones)}")

        if dataset['fatiga'] is not None:
            print(f"   exp_fatigue:   {len(dataset['fatiga'])} registros")
        if dataset['chest'] is not None:
            print(f"   chest:         {len(dataset['chest'])} registros")

        for nombre, df in dataset['wrist'].items():
            if df is not None:
                print(f"   wrist_{nombre}:  {len(df)} registros")

        for banda, df in dataset['eeg'].items():
            if df is not None:
                print(f"   eeg_{banda}:    {len(df)} registros")


# ------------------------------------------------------------------
# Utilidades independientes
# ------------------------------------------------------------------

def convertir_timestamp(
    df: Optional[pd.DataFrame],
    col: str = 'timestamp',
) -> Optional[pd.DataFrame]:
    """Convierte columna de timestamp (ms) a datetime."""
    if df is None:
        return None
    if col in df.columns:
        df = df.copy()
        df['datetime'] = pd.to_datetime(df[col], unit='ms')
    return df