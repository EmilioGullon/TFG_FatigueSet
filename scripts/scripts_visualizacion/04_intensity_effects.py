"""
04_intensity_effects.py
Generar gráficos mostrando efectos de intensidad (Low → Medium → High)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos limpios
df = pd.read_csv(r"c:\Users\egull\OneDrive\Documentos\Proyectos\tfg\scripts_visualizacion\fatigueset_aggregated_features_clean.csv")

print("Generando gráficos de efectos de intensidad...")

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

# Crear figura con 6 subplots (2x3)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Firma de Fatiga Física: Respuesta de Marcadores Fisiológicos a Intensidad\nLow → Medium → High',
             fontsize=14, fontweight='bold', y=0.995)

axes = axes.flatten()

variables = [
    ('hr_media', 'Frecuencia Cardíaca (bpm)', 'HR'),
    ('br_media', 'Tasa Respiratoria (brpm)', 'BR'),
    ('hrv_media', 'Variabilidad HR (ms)', 'HRV'),
    ('eda_media', 'Actividad Electrodermal (µS)', 'EDA'),
    ('eeg_theta_media', 'EEG Theta (Bels)', 'EEG-Theta'),
    ('eeg_alpha_media', 'EEG Alpha (Bels)', 'EEG-Alpha')
]

# Intensidad mapper
intensidad_map = {'low': 1, 'medium': 2, 'high': 3}

for idx, (var, título, abbrev) in enumerate(variables):
    ax = axes[idx]

    # Agrupar por intensidad y calcular medias y desv estándar
    intensidades = []
    medias = []
    stds = []

    for intensidad in ['low', 'medium', 'high']:
        data = df[df['intensidad'] == intensidad][var]
        intensidades.append(intensidad_map[intensidad])
        medias.append(data.mean())
        stds.append(data.std())

    # Plot de línea con error bars
    colors_intensity = ['#2ecc71', '#f39c12', '#e74c3c']
    ax.errorbar(intensidades, medias, yerr=stds, marker='o', markersize=10,
                linewidth=2.5, capsize=5, capthick=2, color='steelblue',
                ecolor='gray', label='Media ± Desv. Estándar')

    # Sombrear áreas por intensidad
    ax.axvspan(0.8, 1.2, alpha=0.1, color='green', label='Baja')
    ax.axvspan(1.8, 2.2, alpha=0.1, color='orange')
    ax.axvspan(2.8, 3.2, alpha=0.1, color='red')

    # Formato
    ax.set_title(f'{título}', fontweight='bold', fontsize=11)
    ax.set_xlabel('Intensidad de Actividad', fontsize=10, fontweight='bold')
    ax.set_ylabel(abbrev, fontsize=10, fontweight='bold')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Baja', 'Media', 'Alta'], fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    # Añadir anotaciones con valores
    for x, y in zip(intensidades, medias):
        ax.annotate(f'{y:.2f}', xy=(x, y), xytext=(0, 10),
                   textcoords='offset points', ha='center', fontsize=8)

plt.tight_layout()

# Guardar
output_file = r"c:\Users\egull\OneDrive\Documentos\Proyectos\tfg\scripts_visualizacion\04_intensity_effects.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Guardado: 04_intensity_effects.png")
plt.close()

# Imprimir análisis
print("\nANÁLISIS DE EFECTOS DE LA INTENSIDAD:")
print("\nCambios esperados según intensidad (firmas de fatiga física):")
print("  ↑ HR incrementa con intensidad")
print("  ↓ HRV disminuye con intensidad (menos variabilidad = más fatiga)")
print("  ↑ EDA aumenta con intensidad (mayor activación)")
print("  ↓ EEG Alpha disminuye (menos relajación)")
print("  ↑ EEG Theta aumenta (señal de fatiga)")

print("\nValores observados en el dataset:")
for var, título, abbrev in variables:
    print(f"\n{abbrev}:")
    for intensidad in ['low', 'medium', 'high']:
        valor = df[df['intensidad'] == intensidad][var].mean()
        print(f"   {intensidad.upper():6} → {valor:.4f}")

print("\nGráficos de intensidad generados exitosamente")
