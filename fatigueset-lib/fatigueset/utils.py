"""
Utilidades para visualización y análisis del dataset FatigueSet.
Basado en las visualizaciones del notebook.
"""
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

COLORES_INTENSIDAD = {
    'low':    '#2196F3',
    'medium': '#FF9800',
    'high':   '#F44336',
}


def plot_fatiga_evolucion(
    df_fatiga: pd.DataFrame,
    participantes: Optional[List[str]] = None,
    figsize: tuple = (16, 5),
) -> plt.Figure:
    """
    Gráfico de evolución de fatiga física y mental (M1→M2→M3)
    con líneas individuales (alpha bajo) y medias por intensidad en negrita.

    Replica la visualización de la Celda 6 del notebook.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    pids = participantes or df_fatiga['participante'].unique()

    for pid in pids:
        for ses in df_fatiga[df_fatiga['participante'] == pid]['sesion'].unique():
            df_s = df_fatiga[
                (df_fatiga['participante'] == pid) &
                (df_fatiga['sesion'] == ses)
            ].sort_values('measurementNumber')
            if df_s.empty:
                continue
            color = COLORES_INTENSIDAD.get(df_s['intensidad'].iloc[0], 'gray')
            axes[0].plot(df_s['measurementNumber'], df_s['physicalFatigueScore'],
                         alpha=0.25, color=color, linewidth=0.8)
            axes[1].plot(df_s['measurementNumber'], df_s['mentalFatigueScore'],
                         alpha=0.25, color=color, linewidth=0.8)

    for intensidad, color in COLORES_INTENSIDAD.items():
        df_i  = df_fatiga[df_fatiga['intensidad'] == intensidad].groupby('measurementNumber')
        mf    = df_i['physicalFatigueScore'].mean()
        mm    = df_i['mentalFatigueScore'].mean()
        axes[0].plot(mf.index, mf.values, color=color, linewidth=3,
                     marker='o', label=f'{intensidad} (media)')
        axes[1].plot(mm.index, mm.values, color=color, linewidth=3,
                     marker='o', label=f'{intensidad} (media)')

    labels = ['M1\n(baseline)', 'M2\n(post-ejercicio)', 'M3\n(post-cognitivo)']
    for ax, titulo in zip(axes, ['Fatiga Física', 'Fatiga Mental']):
        ax.set_title(titulo)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(labels)
        ax.set_ylabel('Score VAS (0-100)')
        ax.legend(fontsize=8)

    fig.suptitle('Evolución de Fatiga — 12 participantes × 3 intensidades',
                 y=1.02, fontsize=13)
    plt.tight_layout()
    return fig


def plot_boxplots_fatigabilidad(
    df_fatigabilidad: pd.DataFrame,
    figsize: tuple = (16, 8),
) -> plt.Figure:
    """
    Boxplots de deltas de fatigabilidad por intensidad.
    Replica la visualización de la Celda 7 del notebook.
    """
    metricas = [
        ('delta_fisica_ejercicio', 'ΔFísica M1→M2 (ejercicio)'),
        ('delta_fisica_cognitivo', 'ΔFísica M2→M3 (cognitivo)'),
        ('delta_fisica_total',     'ΔFísica total M1→M3'),
        ('delta_mental_ejercicio', 'ΔMental M1→M2 (ejercicio)'),
        ('delta_mental_cognitivo', 'ΔMental M2→M3 (cognitivo)'),
        ('delta_mental_total',     'ΔMental total M1→M3'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    for ax, (col, titulo) in zip(axes.flatten(), metricas):
        datos = [df_fatigabilidad[df_fatigabilidad['intensidad'] == i][col].dropna()
                 for i in ['low', 'medium', 'high']]
        bp = ax.boxplot(datos, patch_artist=True, labels=['low', 'medium', 'high'])
        for patch, color in zip(bp['boxes'], COLORES_INTENSIDAD.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(titulo, fontsize=9)
        ax.set_ylabel('Δ Score VAS')

    fig.suptitle('Fatigabilidad por intensidad — N=36 sesiones',
                 y=1.02, fontsize=13)
    plt.tight_layout()
    return fig


def plot_boxplots_sensor(
    df: pd.DataFrame,
    columnas: List[str],
    titulo: str = '',
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """
    Boxplots genéricos de columnas de sensor por intensidad.
    Usado en Celdas 9, 10, 11 del notebook.
    """
    cols_ok = [c for c in columnas if c in df.columns]
    if not cols_ok:
        raise ValueError(f"Ninguna de las columnas {columnas} existe en el DataFrame.")

    figsize = figsize or (5 * len(cols_ok), 4)
    fig, axes = plt.subplots(1, len(cols_ok), figsize=figsize)
    if len(cols_ok) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols_ok):
        datos = [df[df['intensidad'] == i][col].dropna()
                 for i in ['low', 'medium', 'high']]
        bp = ax.boxplot(datos, patch_artist=True, labels=['low', 'medium', 'high'])
        for patch, color in zip(bp['boxes'], COLORES_INTENSIDAD.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(col.upper())

    if titulo:
        fig.suptitle(titulo, y=1.02)
    plt.tight_layout()
    return fig


def plot_correlacion(
    df_ml: pd.DataFrame,
    titulo: str = 'Correlación: Señales Fisiológicas vs Fatiga',
    figsize: tuple = (16, 12),
) -> plt.Figure:
    """
    Heatmap de correlación del dataset ML.
    Replica la visualización de la Celda 16 del notebook.
    """
    cols_num = df_ml.select_dtypes(include=[np.number]).columns.tolist()
    corr = df_ml[cols_num].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f',
        square=True, linewidths=0.5, mask=mask,
        annot_kws={'size': 7}, ax=ax,
    )
    ax.set_title(titulo, fontsize=12)
    plt.tight_layout()
    return fig


def top_correlaciones(
    df_ml: pd.DataFrame,
    target: str = 'fatiga_mental',
    n: int = 10,
) -> pd.Series:
    """
    Devuelve las n features más correlacionadas con target.
    """
    cols_num = df_ml.select_dtypes(include=[np.number]).columns.tolist()
    corr = df_ml[cols_num].corr()
    if target not in corr.columns:
        raise ValueError(f"'{target}' no está en las columnas numéricas del DataFrame.")
    return (
        corr[target]
        .drop([target, 'fatiga_fisica'] if 'fatiga_fisica' in corr.columns else [target],
              errors='ignore')
        .sort_values(key=abs, ascending=False)
        .head(n)
        .round(3)
    )