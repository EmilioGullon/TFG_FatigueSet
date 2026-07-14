"""
02_correlations.py
Generar matriz de correlaciones multimodal con énfasis en patrones de fatiga
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos limpios
df = pd.read_csv(r"c:\Users\egull\OneDrive\Documentos\Proyectos\tfg\scripts_visualizacion\fatigueset_aggregated_features_clean.csv")

print("Generando matriz de correlaciones...")

# Seleccionar variables numéricas relevantes
numeric_cols = ['hr_media', 'br_media', 'hrv_media', 'eda_media',
                'eeg_alpha_media', 'eeg_beta_media', 'eeg_theta_media',
                'fatiga_fisica', 'fatiga_mental']

# Etiquetas descriptivas para la matriz
labels = ['HR (bpm)', 'BR (brpm)', 'HRV (ms)', 'EDA (µS)',
          'EEG Alpha', 'EEG Beta', 'EEG Theta',
          'Fatiga Física', 'Fatiga Mental']

# Calcular matriz de correlación
corr_matrix = df[numeric_cols].corr()

# Crear figura
fig, ax = plt.subplots(figsize=(12, 10))

# Heatmap con anotaciones
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={'label': 'Correlación (r)'},
            xticklabels=labels, yticklabels=labels,
            vmin=-1, vmax=1, ax=ax, cbar=True,
            annot_kws={'size': 8})

# Formato
plt.title('Matriz de Correlación Multimodal - FatigueSet\nCon énfasis en patrones de fatiga',
          fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)

# Ajustar layout
plt.tight_layout()

# Guardar
output_file = r"c:\Users\egull\OneDrive\Documentos\Proyectos\tfg\scripts_visualizacion\02_correlation_matrix.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Guardado: 02_correlation_matrix.png")
plt.close()

# Imprimir algunas correlaciones interesantes con la fatiga
print("\nCorrelaciones más relevantes con FATIGA:")
print("\nCon Fatiga Física:")
fatiga_phys_corr = corr_matrix['fatiga_fisica'].sort_values(ascending=False)
for var, corr in fatiga_phys_corr.items():
    if var != 'fatiga_fisica':
        print(f"   {var:25} : {corr:+.3f}")

print("\nCon Fatiga Mental:")
fatiga_ment_corr = corr_matrix['fatiga_mental'].sort_values(ascending=False)
for var, corr in fatiga_ment_corr.items():
    if var != 'fatiga_mental':
        print(f"   {var:25} : {corr:+.3f}")

print("\nMatriz de correlaciones generada exitosamente")
