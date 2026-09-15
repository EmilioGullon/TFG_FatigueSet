import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Style setup
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'

csv_path = Path('Jupyters/1.Preprocesado/fatigueset_aggregated_features.csv')
if not csv_path.exists():
    csv_path = Path('scripts/scripts_visualizacion/fatigueset_aggregated_features_clean.csv')

df_ml = pd.read_csv(csv_path)

# Global Z-score
mu_global = df_ml['hr_media'].mean()
std_global = df_ml['hr_media'].std()
df_global = df_ml.copy()
df_global['hr_norm'] = (df_ml['hr_media'] - mu_global) / std_global

# Per-subject Z-score
df_sujeto = df_ml.copy()
df_sujeto['hr_norm'] = 0.0
for pid, grp in df_ml.groupby('participante'):
    mu_p = grp['hr_media'].mean()
    std_p = grp['hr_media'].std()
    if std_p == 0 or np.isnan(std_p):
        std_p = 1.0
    df_sujeto.loc[grp.index, 'hr_norm'] = (grp['hr_media'] - mu_p) / std_p

participants = sorted(df_ml['participante'].unique())
x_labels = [f"P{p:02d}" if isinstance(p, int) else str(p) for p in participants]

data_orig = [df_ml[df_ml['participante'] == p]['hr_media'].dropna().values for p in participants]
data_glob = [df_global[df_global['participante'] == p]['hr_norm'].dropna().values for p in participants]
data_suj = [df_sujeto[df_sujeto['participante'] == p]['hr_norm'].dropna().values for p in participants]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3), dpi=300)

configs = [
    {
        'ax': axes[0],
        'data': data_orig,
        'title': '(A) Bioseñales en Crudo',
        'subtitle': 'Línea base dispar entre sujetos (60 - 100+ bpm)',
        'ylabel': 'Frecuencia Cardíaca (bpm)',
        'box_face': '#F1F5F9',     # Slate-100
        'box_edge': '#475569',     # Slate-600
        'median_col': '#DC2626',   # Red median
        'hline': mu_global,
        'hline_label': f'Media global = {mu_global:.1f} bpm',
        'hline_col': '#DC2626',
        'ylim': (50, 175),
        'tag_color': '#475569'
    },
    {
        'ax': axes[1],
        'data': data_glob,
        'title': '(B) Fallo de Z-Score Global',
        'subtitle': 'Preserva el sesgo individual intacto',
        'ylabel': 'Z-Score Global',
        'box_face': '#FFE4E6',     # Rose-100
        'box_edge': '#E11D48',     # Rose-600
        'median_col': '#BE123C',   # Rose-700
        'hline': 0.0,
        'hline_label': r'Referencia $\mu_{global} = 0$',
        'hline_col': '#E11D48',
        'ylim': (-1.8, 4.3),
        'tag_color': '#E11D48'
    },
    {
        'ax': axes[2],
        'data': data_suj,
        'title': '(C) Z-Score Intra-Sujeto (Éxito)',
        'subtitle': 'Todos los sujetos alineados en el origen',
        'ylabel': 'Z-Score Intra-Sujeto',
        'box_face': '#E0F2FE',     # Sky-100
        'box_edge': '#0284C7',     # Sky-600
        'median_col': '#0369A1',   # Sky-700
        'hline': 0.0,
        'hline_label': r'Referencia $\mu_{i} = 0$',
        'hline_col': '#0284C7',
        'ylim': (-1.8, 2.9),
        'tag_color': '#0284C7'
    }
]

for cfg in configs:
    ax = cfg['ax']
    data = cfg['data']
    
    bp = ax.boxplot(
        data,
        patch_artist=True,
        tick_labels=x_labels,
        widths=0.60,
        showmeans=False,
        medianprops=dict(color=cfg['median_col'], linewidth=2.2),
        whiskerprops=dict(color=cfg['box_edge'], linewidth=1.2),
        capprops=dict(color=cfg['box_edge'], linewidth=1.2),
        flierprops=dict(marker='o', markersize=3.8, markerfacecolor='white', markeredgecolor=cfg['box_edge'], alpha=0.8)
    )
    
    for patch in bp['boxes']:
        patch.set_facecolor(cfg['box_face'])
        patch.set_edgecolor(cfg['box_edge'])
        patch.set_linewidth(1.3)
        
    ax.axhline(cfg['hline'], color=cfg['hline_col'], linestyle='--', linewidth=1.5, alpha=0.9, label=cfg['hline_label'])
    
    # Title with bold main title and subtitle below
    ax.set_title(f"{cfg['title']}\n{cfg['subtitle']}", fontsize=11, fontweight='bold', pad=8, color='#0F172A', linespacing=1.25)
    ax.set_xlabel('Participante', fontsize=10, fontweight='semibold', color='#334155', labelpad=4)
    ax.set_ylabel(cfg['ylabel'], fontsize=10, fontweight='semibold', color='#334155')
    
    ax.set_ylim(cfg['ylim'])
    ax.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')
    ax.tick_params(axis='x', labelsize=8.5, rotation=0)
    ax.tick_params(axis='y', labelsize=8.5)
    ax.set_axisbelow(True)
    
    ax.legend(loc='upper right', fontsize=8.5, framealpha=0.92, facecolor='white', edgecolor='#E2E8F0')

plt.tight_layout()

out_path = Path('presentacion/figures/normalizacion_comparativa_sujeto_hr_beamer.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved: {out_path}")
