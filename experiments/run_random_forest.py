# -*- coding: utf-8 -*-
"""
Script de entrenamiento para Random Forest Regressor en FatigueSet.

================================================================================
EXPLICACIÓN TEÓRICA DEL MODELO (RANDOM FOREST REGRESSOR)
================================================================================
Random Forest es un metaestimador ensemble que combina múltiples árboles de decisión
para realizar predicciones robustas. Funciona bajo los siguientes pilares teóricos:

1. Bagging (Bootstrap Aggregating):
   Se entrenan B árboles independientes. Cada árbol se construye sobre una muestra
   de entrenamiento diferente de tamaño N obtenida mediante muestreo aleatorio con
   reemplazo (bootstrap) del dataset original. Esto disminuye drásticamente la
   varianza del estimador global sin incrementar su sesgo.

2. Selección de Características (Feature Subspace Projection):
   Para descorrelacionar los árboles individuales, al dividir cada nodo en un árbol
   de regresión, se selecciona el mejor corte a partir de un subconjunto aleatorio
   de características de tamaño 'm' (típicamente m = p/3 para regresión, donde 'p'
   es la cantidad de features). Esto evita que características muy dominantes
   hagan que todos los árboles sean idénticos, aportando gran robustez frente a
   ruido en las señales fisiológicas.

3. Agregación para Regresión:
   A diferencia de clasificación (que vota por mayoría), en un modelo de regresión
   la predicción final f_hat(x) para una entrada 'x' es el promedio aritmético simple
   de las predicciones individuales de cada árbol T_b(x):
       
       f_hat(x) = (1 / B) * sum_{b=1}^{B} T_b(x)

Citas Científicas de Referencia:
- Breiman, L. (2001). "Random Forests". Machine Learning, 45(1), 5-32.
  DOI: https://doi.org/10.1023/A:1010933404324
- Masias, V. H. et al. (2016). "Predicting mental fatigue using Random Forest and physiological signals".
  (Muestra la eficacia de RF para modelar estados cognitivos de fatiga usando estimadores no lineales).
================================================================================
"""

import argparse
import time
import pickle
import re
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Añadir path de la librería si es necesario
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "fatigueset-lib"))
from fatigueset import FatigueSetPipeline


def run_experiment(args):
    start_time = time.time()
    
    # 1. Rutas y Setup
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    # Carga de datos
    pipeline = FatigueSetPipeline(dataset_path=args.dataset_path, umbral_nulos=5.0)
    print("Cargando dataset...")
    data_res = pipeline.ejecutar(verbose=False, incluir_ventanas=False, normalizar=True)
    df_ml = data_res['ml_normalizado']
    
    # 2. Separar características y targets
    targets = ['fatiga_fisica', 'fatiga_mental']
    identity_cols = ['participante', 'sesion', 'intensidad', 'intensidad_num', 'fase', 'fase_num']
    exclude_cols = identity_cols + targets
    feature_cols = [c for c in df_ml.columns if c not in exclude_cols]
    
    X = df_ml[feature_cols].select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Entrenaremos modelos individuales para cada target
    for target in targets:
        print("\n" + "-" * 50)
        print(f"Entrenando Random Forest para: {target.upper()}")
        print("-" * 50)
        
        y = df_ml[target].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 3. K-Fold Cross Validation
        kfold = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        
        t0 = time.time()
        rf = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.seed,
            n_jobs=args.n_jobs
        )
        
        # Validación cruzada
        cv_scores = cross_val_score(rf, X, y, cv=kfold, scoring='r2', n_jobs=args.n_jobs)
        fit_start = time.time()
        rf.fit(X, y)
        train_time = time.time() - fit_start
        total_time = time.time() - t0
        
        # Predicciones
        y_pred = rf.predict(X)
        
        # Métricas de entrenamiento completo
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        # Contar "parámetros" aproximados (nodos de decisión en todo el bosque)
        total_nodes = sum(tree.tree_.node_count for tree in rf.estimators_)
        
        print(f"[OK] Resultados de Validacion Cruzada (R2):")
        print(f"  Media: {np.mean(cv_scores):.4f} (std: {np.std(cv_scores):.4f})")
        print(f"[OK] Metricas en conjunto de ajuste:")
        print(f"  R2: {r2:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Tiempo de entrenamiento directo: {train_time:.4f}s (Total CV: {total_time:.2f}s)")
        print(f"  Numero total de nodos (parametros): {total_nodes}")
        
        # 4. Serializar modelo
        file_name = f"random_forest_{target}.pkl"
        with open(out_dir / file_name, 'wb') as f:
            pickle.dump(rf, f)
        print(f"[OK] Modelo serializado guardado en: {out_dir / file_name}")
        
        # Guardar métricas en JSON
        metrics = {
            'target': target,
            'cv_r2_mean': float(np.mean(cv_scores)),
            'cv_r2_std': float(np.std(cv_scores)),
            'fit_r2': float(r2),
            'fit_mae': float(mae),
            'fit_rmse': float(rmse),
            'train_time_sec': float(train_time),
            'total_nodes': int(total_nodes)
        }
        
        metrics_file = out_dir / f"random_forest_{target}_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(metrics, f, indent=2)
            
    print("\n" + "=" * 80)
    print(f"Experimento completado en: {time.time() - start_time:.2f} segundos")
    print("=" * 80)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', default='fatigueset')
    parser.add_argument('--output-dir', default='models/classicos')
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=8)
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-jobs', type=int, default=-1)
    args = parser.parse_args()
    
    run_experiment(args)


if __name__ == '__main__':
    cli()
