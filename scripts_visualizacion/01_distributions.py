"""
01_distributions.py
Generar gráficos de distribuciones para las 6 variables fisiológicas clave
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos limpios
df = pd.read_csv(r"c:\Users\egull\OneDrive\Documentos\Proyectos\tfg\scripts_visualizacion\fatigueset_aggregated_features_clean.csv")

print("Generando gráficos de distribuciones...")

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

# Crear figura con 6 subplots (2x3)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Distribuciones de Variables Fisiológicas - FatigueSet',
             fontsize=16, fontweight='bold', y=0.995)

axes = axes.flatten()

variables = [
    ('hr_media', 'Frecuencia Cardíaca (bpm)', 'bpm'),
    ('br_media', 'Tasa Respiratoria (brpm)', 'brpm'),
    ('hrv_media', 'Variabilidad Frecuencia Cardíaca (ms)', 'ms'),
    ('eda_media', 'Actividad Electrodermal (µS)', 'µS'),
    ('eeg_alpha_media', 'EEG Alpha (Bels)', 'Bels'),
    ('eeg_beta_media', 'EEG Beta (Bels)', 'Bels')
]

for idx, (var, título, unidad) in enumerate(variables):
    ax = axes[idx]
    data = df[var].dropna()

    # Histograma
    ax.hist(data, bins=25, density=True, alpha=0.6, color='steelblue', edgecolor='black')

    # KDE (línea suave de densidad)
    data.plot.kde(ax=ax, linewidth=2, color='darkred', label='KDE')

    # Etiquetas y formato
    ax.set_title(f'{título}', fontweight='bold', fontsize=11)
    ax.set_xlabel(unidad, fontsize=10)
    ax.set_ylabel('Densidad', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(['Densidad (KDE)', 'Histograma'], loc='upper right', fontsize=9)

    # Añadir estadísticas
    stats_text = f'µ = {data.mean():.2f}\nσ = {data.std():.2f}\nn = {len(data)}'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=8)

plt.tight_layout()
output_file = r"c:\Users\egull\OneDrive\Documentos\Proyectos\tfg\scripts_visualizacion\01_distributions.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Guardado: 01_distributions.png")
plt.close()

print("Distribuciones generadas exitosamente")
