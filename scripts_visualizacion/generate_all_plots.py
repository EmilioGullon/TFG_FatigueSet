import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv(r"c:\Users\egull\OneDrive\Documentos\Proyectos\tfg\scripts_visualizacion\fatigueset_aggregated_features_clean.csv")

print("Generating visualizations...")

# 1. DISTRIBUTIONS
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Distribuciones de Variables Fisiologicas - FatigueSet',
             fontsize=16, fontweight='bold', y=0.995)

vars_list = [
    ('hr_media', 'Frecuencia Cardiaca (bpm)'),
    ('br_media', 'Tasa Respiratoria (brpm)'),
    ('hrv_media', 'Variabilidad HR (ms)'),
    ('eda_media', 'Actividad Electrodermal (uS)'),
    ('eeg_alpha_media', 'EEG Alpha (Bels)'),
    ('eeg_beta_media', 'EEG Beta (Bels)')
]

axes_flat = axes.flatten()
for idx, (var, title) in enumerate(vars_list):
    ax = axes_flat[idx]
    data = df[var].dropna()
    ax.hist(data, bins=25, density=True, alpha=0.6, color='steelblue', edgecolor='black')
    data.plot.kde(ax=ax, linewidth=2, color='darkred', label='KDE')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Valor')
    ax.set_ylabel('Densidad')
    ax.grid(True, alpha=0.3)
    ax.legend(['Densidad (KDE)'], loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(r"c:\Users\egull\OneDrive\Documentos\Proyectos\tfg\scripts_visualizacion\01_distributions.png", dpi=300, bbox_inches='tight')
print("OK: 01_distributions.png")
plt.close()

# 2. CORRELATION MATRIX
numeric_cols = ['hr_media', 'br_media', 'hrv_media', 'eda_media',
                'eeg_alpha_media', 'eeg_beta_media', 'eeg_theta_media',
                'fatiga_fisica', 'fatiga_mental']

corr_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={'label': 'Correlacion'},
            vmin=-1, vmax=1, ax=ax)
plt.title('Matriz de Correlacion Multimodal', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(r"c:\Users\egull\OneDrive\Documentos\Proyectos\tfg\scripts_visualizacion\02_correlation_matrix.png", dpi=300, bbox_inches='tight')
print("OK: 02_correlation_matrix.png")
plt.close()

# 3. PHASE CHANGES
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Cambios a traves de Fases Experimentales',
             fontsize=14, fontweight='bold')

fase_order = ['M1_baseline', 'M2_post_ejercicio', 'M3_post_fatiga_mental']

for idx, (var, title) in enumerate(vars_list):
    ax = axes.flatten()[idx]
    sns.boxplot(data=df, x='fase', y=var, hue='intensidad',
                order=fase_order,
                hue_order=['low', 'medium', 'high'],
                palette={'low': '#2ecc71', 'medium': '#f39c12', 'high': '#e74c3c'},
                ax=ax, width=0.7)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Fase')
    ax.set_ylabel('Valor')
    ax.legend().remove() if idx > 0 else None

plt.tight_layout()
plt.savefig(r"c:\Users\egull\OneDrive\Documentos\Proyectos\tfg\scripts_visualizacion\03_phase_changes.png", dpi=300, bbox_inches='tight')
print("OK: 03_phase_changes.png")
plt.close()

# 4. INTENSITY EFFECTS
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Firma de Fatiga Fisica: Efectos de Intensidad',
             fontsize=14, fontweight='bold')

for idx, (var, title) in enumerate(vars_list):
    ax = axes.flatten()[idx]

    intensidades = []
    medias = []
    stds = []

    for i, int_val in enumerate(['low', 'medium', 'high']):
        data = df[df['intensidad'] == int_val][var]
        intensidades.append(i+1)
        medias.append(data.mean())
        stds.append(data.std())

    ax.errorbar(intensidades, medias, yerr=stds, marker='o', markersize=10,
                linewidth=2.5, capsize=5, color='steelblue')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Intensidad')
    ax.set_ylabel('Valor')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Baja', 'Media', 'Alta'])
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r"c:\Users\egull\OneDrive\Documentos\Proyectos\tfg\scripts_visualizacion\04_intensity_effects.png", dpi=300, bbox_inches='tight')
print("OK: 04_intensity_effects.png")
plt.close()

# 5. INDIVIDUAL VARIABILITY
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Variabilidad Entre Participantes',
             fontsize=14, fontweight='bold')

for idx, (var, title) in enumerate(vars_list):
    ax = axes.flatten()[idx]
    sns.violinplot(data=df, x='participante', y=var, ax=ax, palette='Set2')
    mean_all = df[var].mean()
    ax.axhline(mean_all, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Participante')
    ax.set_ylabel('Valor')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(r"c:\Users\egull\OneDrive\Documentos\Proyectos\tfg\scripts_visualizacion\05_individual_variability.png", dpi=300, bbox_inches='tight')
print("OK: 05_individual_variability.png")
plt.close()

print("\nAll visualizations generated successfully!")
