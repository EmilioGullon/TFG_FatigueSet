import pandas as pd
import numpy as np
import warnings
import sys

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

data_path = r"c:\Users\egull\OneDrive\Documentos\Proyectos\tfg\fatigueset_aggregated_features.csv"

print("="*70)
print("CARGA Y LIMPIEZA DE DATOS FATIGUESET")
print("="*70)

print("\n** Cargando datos desde: fatigueset_aggregated_features.csv")
df = pd.read_csv(data_path)
print(f"   Registros originales: {len(df)}")
print(f"   Columnas: {len(df.columns)}")

print("\nEstructura de datos:")
print(f"   Participantes: {df['participante'].nunique()}")
print(f"   Fases: {list(df['fase'].unique())}")
print(f"   Intensidades: {list(df['intensidad'].unique())}")

print("\nProcesando valores especiales...")
df_clean = df.copy()
inf_count = np.isinf(df_clean.select_dtypes(include=[np.number])).sum().sum()
nan_count = df_clean.select_dtypes(include=[np.number]).isna().sum().sum()
print(f"   Valores inf encontrados: {inf_count}")
print(f"   Valores NaN encontrados: {nan_count}")

df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
key_vars = ['eeg_alpha_media', 'eeg_beta_media', 'eeg_theta_media',
            'hr_media', 'br_media', 'hrv_media', 'eda_media']
df_clean = df_clean.dropna(subset=key_vars)
print(f"   Registros despues de limpieza: {len(df_clean)}")

# Guardar datos limpios
output_path = r"c:\Users\egull\OneDrive\Documentos\Proyectos\tfg\scripts_visualizacion\fatigueset_aggregated_features_clean.csv"
df_clean.to_csv(output_path, index=False)
print(f"\nDatos guardados: fatigueset_aggregated_features_clean.csv")
print(f"Total registros: {len(df_clean)}")
print("\n" + "="*70)
print("LISTO PARA PROCESAR VISUALIZACIONES")
print("="*70)
