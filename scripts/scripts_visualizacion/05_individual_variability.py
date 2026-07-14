"""
05_individual_variability.py
Generar violin plots mostrando variabilidad entre participantes
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos limpios
df = pd.read_csv(r"c:\Users\egull\OneDrive\Documentos\Proyectos\tfg\scripts_visualizacion\fatigueset_aggregated_features_clean.csv")

print("Generando gráficos de variabilidad individual...")

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 9

# Crear figura con 6 subplots (2x3)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Variabilidad Entre Participantes: Diferencias Individuales en Susceptibilidad a Fatiga',
             fontsize=14, fontweight='bold', y=0.995)

axes = axes.flatten()

variables = [
    ('hr_media', 'Frecuencia Cardíaca (bpm)', 'HR'),
    ('br_media', 'Tasa Respiratoria (brpm)', 'BR'),
    ('hrv_media', 'Variabilidad HR (ms)', 'HRV'),
    ('eda_media', 'Actividad Electrodermal (µS)', 'EDA'),
    ('eeg_alpha_media', 'EEG Alpha (Bels)', 'EEG-Alpha'),
    ('eeg_beta_media', 'EEG Beta (Bels)', 'EEG-Beta')
]

for idx, (var, título, abbrev) in enumerate(variables):
    ax = axes[idx]

    # Violin plot con puntos individuales
    sns.violinplot(data=df, x='participante', y=var, ax=ax,
                   inner='quartile', palette='Set2')
    sns.stripplot(data=df, x='participante', y=var, ax=ax,
                  size=3, color='black', alpha=0.4, jitter=True)

    # Formato
    ax.set_title(f'{título}', fontweight='bold', fontsize=11)
    ax.set_xlabel('Participante', fontsize=10, fontweight='bold')
    ax.set_ylabel(abbrev, fontsize=10, fontweight='bold')
    ax.tick_params(axis='x', rotation=0, labelsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Línea horizontal para media general
    mean_all = df[var].mean()
    ax.axhline(mean_all, color='red', linestyle='--', linewidth=1.5,
              alpha=0.7, label=f'Media general = {mean_all:.2f}')
    ax.legend(fontsize=7, loc='best')

plt.tight_layout()

# Guardar
output_file = r"c:\Users\egull\OneDrive\Documentos\Proyectos\tfg\scripts_visualizacion\05_individual_variability.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Guardado: 05_individual_variability.png")
plt.close()

# Análisis de variabilidad
print("\nANÁLISIS DE VARIABILIDAD INDIVIDUAL:")
print("\nDesviación estándar entre participantes (heterogeneidad cardiovascular/fisiológica):")

for var, título, abbrev in variables:
    print(f"\n{abbrev} - {título}:")
    por_participante = df.groupby('participante')[var].mean().sort_values()

    media_gral = por_participante.mean()
    desv_gral = por_participante.std()

    print(f"   Media entre participantes: {media_gral:.4f}")
    print(f"   Desv. Est. entre participantes: {desv_gral:.4f} ({desv_gral/media_gral*100:.1f}%)")
    print(f"   Rango: [{por_participante.min():.4f}, {por_participante.max():.4f}]")

    # Participantes extremos
    print(f"   Máximo: P{por_participante.idxmax()} ({por_participante.max():.4f})")
    print(f"   Mínimo: P{por_participante.idxmin()} ({por_participante.min():.4f})")

print("\nGráficos de variabilidad generados exitosamente")
