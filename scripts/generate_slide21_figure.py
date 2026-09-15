#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador del gráfico de comparativa MAE para modelos Deep Learning (Diapositiva 21).
Utiliza los datos oficiales exactos de la memoria y la paleta Slate & Modern Data Science.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Configuración de estilo
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Datos oficiales de la memoria
models = [
    'Custom GRU',
    'Custom xLSTM (sLSTM)',
    'Custom LSTM',
    'Custom CNN-LSTM',
    'Custom TCN Dilatada',
    'Custom Transformer',
    'Custom PatchTST',
    'RNN Estándar (Baseline)'
]

mae_global = [16.99, 17.33, 17.50, 17.56, 17.70, 18.25, 19.44, 30.09]
mae_fisica = [16.12, 16.48, 16.65, 16.70, 16.95, 17.40, 18.20, 29.50]
mae_mental = [17.86, 18.18, 18.35, 18.42, 18.45, 19.10, 20.68, 30.68]

# Invertir para que el mejor quede arriba
models = models[::-1]
mae_global = mae_global[::-1]
mae_fisica = mae_fisica[::-1]
mae_mental = mae_mental[::-1]

# Colores según categoría/rendimiento
# Slate & Modern Data Science palette
colors = [
    '#e11d48',  # RNN Básica (Rojo/Coral)
    '#f59e0b',  # PatchTST (Ámbar)
    '#d97706',  # Transformer (Ámbar oscuro)
    '#0284c7',  # TCN (Azul)
    '#0284c7',  # CNN-LSTM (Azul)
    '#0369a1',  # LSTM (Azul profundo)
    '#6366f1',  # xLSTM (Índigo moderno)
    '#10b981',  # GRU (Verde esmeralda - Campeón)
]

fig, ax = plt.subplots(figsize=(10, 5.8))
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#f8fafc')

y_pos = np.arange(len(models))
bars = ax.barh(y_pos, mae_global, color=colors, height=0.62, edgecolor='#0f172a', linewidth=0.8, zorder=3)

# Línea vertical de referencia del mejor modelo (GRU = 16.99)
ax.axvline(16.99, color='#10b981', linestyle='--', linewidth=1.5, alpha=0.85, zorder=2,
           label='Mejor MAE Global (GRU = 16.99)')

# Anotaciones numéricas con detalle de Física y Mental
for bar, mg, mf, mm in zip(bars, mae_global, mae_fisica, mae_mental):
    w = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2.0
    text = f" {mg:.2f}  " + r"$\mathbf{(F\acute{\imath}s:\ " + f"{mf:.1f}" + r"\ |\ Ment:\ " + f"{mm:.1f}" + r")}$"
    ax.text(w + 0.3, y, text, va='center', ha='left', fontsize=9.5, color='#0f172a', fontweight='normal')

# Títulos y ejes
ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontweight='bold', color='#0f172a')
ax.set_xlabel('Error Absoluto Medio (MAE en Escala VAS 0–100)', fontweight='bold', color='#0f172a', labelpad=8)
ax.set_title('Rendimiento Comparativo de Modelos Deep Learning en Validación GroupKFold\n'
             r'$\bf{Sesgo\ Inductivo\ Recurrente\ frente\ a\ Arquitecturas\ Convolucionales\ y\ Atencionales}$',
             fontweight='bold', fontsize=12, color='#0f172a', pad=12)

ax.set_xlim(0, 36)
ax.grid(axis='x', linestyle=':', alpha=0.6, color='#94a3b8', zorder=0)
ax.set_axisbelow(True)

# Spines styling
for spine in ax.spines.values():
    spine.set_color('#cbd5e1')
    spine.set_linewidth(1.0)

# Leyenda
ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=9.5)

plt.tight_layout()

# Guardar en presentacion/figures
out_dir = os.path.join(os.path.dirname(__file__), '..', 'presentacion', 'figures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'comparativa_mae_deep.png')
plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print(f"[OK] Gráfico generado exitosamente en: {out_path}")
