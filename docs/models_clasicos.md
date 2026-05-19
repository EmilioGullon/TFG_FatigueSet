# Modelos Clásicos — Documentación y Protocolo

Este documento describe en detalle el proceso, los modelos, las métricas y las pruebas realizadas para los modelos clásicos implementados en `Jupyters/modelos_clasicos.ipynb` y `experiments/run_models_classicos.py`.

## Resumen
- Objetivo: predecir `fatiga_fisica` y `fatiga_mental` desde features fisiológicas y de tarea.
- Tipo de problema: regresión multivariada (en la práctica tratamos cada target por separado, como en el notebook).

## Datos
- El dataset real no se incluye en el repositorio por motivos de privacidad y tamaño.
- Para reproducibilidad ligera se incluye `fatigueset-lib/data/sample/sample_df.csv` con un subconjunto sintético/ejemplo.

## Preprocesado
- Selección de columnas numéricas como features.
- Relleno de nulos con 0 en el script de ejemplo (para evitar fallos en tests); en el flujo real usar imputación adecuada.
- Normalización y ventanas se realizan en el pipeline `fatigueset-lib` si se usa con datos reales.

## Modelos considerados
- KNN (k=3, k=5)
- Regresión Lineal
- Ridge, Lasso, ElasticNet
- Decision Tree (max_depth=5)
- Random Forest (n_estimators=50, max_depth=5)
- SVM (RBF)

Los hiperparámetros se han elegido para ser representativos y rápidos; para un estudio completo se recomienda GridSearchCV/RandomizedSearchCV.

## Protocolo experimental
- Validación: KFold 5-fold, shuffle=True, seed=42.
- Métricas: CV R² (media y desviación), MSE, MAE, R² entrenado.
- Visualizaciones: barras de CV R² con error, barras de MAE, boxplots comparativos.

## Resultados y comparaciones
- Ejecuta `python experiments/run_models_classicos.py --data-path fatigueset-lib/data/sample/sample_df.csv --output-dir output/example` para generar `results_models.csv`, `cv_scores.pkl` y figuras.
- Ejecuta `python experiments/compare_models.py` para generar comparaciones estadísticas (bootstrap top-2) y figura `compare_models.png`.

Interpretación: dado el tamaño reducido del conjunto de ejemplo, las métricas son informativas solo a modo de prueba; con el dataset real (108 muestras) las conclusiones deben tomarse con cautela debido a la varianza de CV.

## Reproducibilidad
- Comandos rápidos:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r fatigueset-lib/requirements.txt
python experiments/run_models_classicos.py --data-path fatigueset-lib/data/sample/sample_df.csv --output-dir output/example
python experiments/compare_models.py
```

## Pruebas automáticas
- `pytest -q` ejecutará las pruebas unitarias y e2e que usan el CSV de ejemplo.

## Limitaciones
- Dataset real pequeño → variabilidad alta en estimadores.
- Modelos no han sido hiperparametrizados exhaustivamente.

## Próximos pasos
- Añadir GridSearchCV/RandomizedSearchCV para tunear hiperparámetros.
- Incluir evaluación por subject holdout para estimar generalización real.
- Probar métodos de ensamblado y calibración de intervalos de predicción.
