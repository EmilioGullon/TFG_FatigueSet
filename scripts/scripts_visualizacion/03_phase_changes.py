"""
03_phase_changes.py
Generar box plots mostrando cambios entre fases experimentales
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos limpios
df = pd.read_csv(r"c:\Users\egull\OneDrive\Documentos\Proyectos\tfg\scripts_visualizacion\fatigueset_aggregated_features_clean.csv")

print("Generando gráficos de cambios por fase...")

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

# Crear figura del 6 subplots (2x3)
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Cambios de Variables Fisiológicas a través de Fases Experimentales\nM1 (Baseline) → M2 (Post-Ejercicio) → M3 (Post-Fatiga Mental)',
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

# Mapper para orden de fases
fase_order = ['M1_baseline', 'M2_post_ejercicio', 'M3_post_fatiga_mental']
fase_labels = ['M1\nBaseline', 'M2\nPost-Ejercicio', 'M3\nPost-Fatiga']

for idx, (var, título, abbrev) in enumerate(variables):
    ax = axes[idx]

    # Crear box plot con hue por intensidad
    sns.boxplot(data=df, x='fase', y=var, hue='intensidad',
                order=fase_order,
                hue_order=['low', 'medium', 'high'],
                palette={'low': '#2ecc71', 'medium': '#f39c12', 'high': '#e74c3c'},
                ax=ax, width=0.7)

    # Formato
    ax.set_title(f'{título}', fontweight='bold', fontsize=11)
    ax.set_xlabel('Fase Experimental', fontsize=10, fontweight='bold')
    ax.set_ylabel(abbrev, fontsize=10, fontweight='bold')
    ax.set_xticklabels(fase_labels, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Leyenda
    handles, labels_legend = ax.get_legend_handles_labels()
    if idx == 0:  # Solo mostrar leyenda en el primer gráfico
        ax.legend(handles[3:], ['Baja', 'Media', 'Alta'],
                  title='Intensidad', title_fontsize=9, fontsize=8,
                  loc='upper left')
    else:
        ax.legend().remove()

plt.tight_layout()

# Guardar
output_file = r"c:\Users\egull\OneDrive\Documentos\Proyectos\tfg\scripts_visualizacion\03_phase_changes.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Guardado: 03_phase_changes.png")
plt.close()

# Imprimir resumen de cambios
print("\nRESUMEN DE CAMBIOS ENTRE FASES:")
for var, _, abbrev in variables:
    print(f"\n{abbrev}:")
    for intensity in ['low', 'medium', 'high']:
        m1 = df[(df['fase'] == 'M1_baseline') & (df['intensidad'] == intensity)][var].mean()
        m2 = df[(df['fase'] == 'M2_post_ejercicio') & (df['intensidad'] == intensity)][var].mean()
        m3 = df[(df['fase'] == 'M3_post_fatiga_mental') & (df['intensidad'] == intensity)][var].mean()
        change_m1_m2 = ((m2 - m1) / m1 * 100) if m1 != 0 else 0
        change_m2_m3 = ((m3 - m2) / m2 * 100) if m2 != 0 else 0
        print(f"   {intensity.upper():6} - M1→M2: {change_m1_m2:+.1f}%,  M2→M3: {change_m2_m3:+.1f}%")

print("\nGráficos de fases generados exitosamente")
