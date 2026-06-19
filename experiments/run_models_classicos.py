"""Run Models Clásicos - script reproducible
Genera `output/results_models.csv`, guarda `cv_scores.pkl` y figuras en `output/`.
"""
import argparse
from pathlib import Path
import json
import pickle
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def run(args):
    data_path = Path(args.data_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)

    # identity and targets
    targets = ['fatiga_fisica', 'fatiga_mental']
    identity_cols = ['participante', 'sesion', 'intensidad', 'intensidad_num', 'fase', 'fase_num']
    exclude_cols = identity_cols + targets
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    y = df['fatiga_fisica']

    kfold = KFold(n_splits=5, shuffle=True, random_state=args.seed)

    modelos = {
        'KNN (k=3)': KNeighborsRegressor(n_neighbors=3),
        'KNN (k=5)': KNeighborsRegressor(n_neighbors=5),
        'Regresión Lineal': LinearRegression(),
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Lasso (α=0.1)': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1),
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=args.seed),
        'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=args.seed),
        'SVM': SVR(kernel='rbf', C=100),
    }

    resultados = {}
    cv_scores_all = {}

    for nombre, modelo in modelos.items():
        cv_scores = cross_val_score(modelo, X, y, cv=kfold, scoring='r2', n_jobs=args.n_jobs)
        cv_scores_all[nombre] = cv_scores.tolist()

        modelo.fit(X, y)
        y_pred = modelo.predict(X)

        # Serializar y guardar el modelo entrenado
        nombre_sanitizado = re.sub(r'[^a-zA-Z0-9_]', '_', nombre.lower().replace(' ', '_'))
        with open(out_dir / f"{nombre_sanitizado}.pkl", 'wb') as f:
            pickle.dump(modelo, f)

        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        resultados[nombre] = {
            'cv_r2_mean': float(np.mean(cv_scores)),
            'cv_r2_std': float(np.std(cv_scores)),
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2),
        }

    df_results = pd.DataFrame(resultados).T
    df_results = df_results.sort_values('cv_r2_mean', ascending=False)
    df_results.to_csv(out_dir / 'results_models.csv')

    # Guardar cv_scores
    with open(out_dir / 'cv_scores.pkl', 'wb') as f:
        pickle.dump(cv_scores_all, f)

    # Figura resumen: CV R2 y MAE
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    df_plot = df_results.sort_values('cv_r2_mean')
    axes[0].barh(df_plot.index, df_plot['cv_r2_mean'], xerr=df_plot['cv_r2_std'], color='steelblue')
    axes[0].set_title('CV R² (5-fold)')
    axes[0].set_xlabel('R²')

    df_plot_mae = df_results.sort_values('mae')
    axes[1].barh(df_plot_mae.index, df_plot_mae['mae'], color='coral')
    axes[1].set_title('MAE (entrenamiento)')
    axes[1].set_xlabel('MAE')

    plt.tight_layout()
    fig.savefig(out_dir / 'summary_models.png')

    # Feature importances si Random Forest
    if 'Random Forest' in modelos:
        rf = modelos['Random Forest']
        if hasattr(rf, 'feature_importances_'):
            importancias = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            importancias.to_csv(out_dir / 'feature_importances.csv', index=False)

    # Evitar caracteres no soportados por algunas consolas Windows
    print(f"Resultados guardados en: {out_dir}")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='fatigueset-lib/data/sample/sample_df.csv')
    parser.add_argument('--output-dir', default='models/classicos')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-jobs', type=int, default=1)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    cli()
