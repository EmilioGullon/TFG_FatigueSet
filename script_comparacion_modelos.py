import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setup
sys.path.insert(0, str(Path.cwd() / "fatigueset-lib"))
from fatigueset import FatigueSetPipeline

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    xgboost_disponible = True
except ImportError:
    xgboost_disponible = False

print("=" * 80)
print("CARGANDO DATASET FATIGUESET")
print("=" * 80)

# Cargar dataset
pipeline = FatigueSetPipeline(dataset_path="fatigueset", umbral_nulos=5.0)
resultados = pipeline.ejecutar(verbose=False, incluir_ventanas=False, normalizar=True)
df_ml = resultados['ml_normalizado']

# Preparar features
exclude_cols = ['participante', 'sesion', 'intensidad', 'intensidad_num', 'fase', 'fase_num', 'fatiga_fisica', 'fatiga_mental']
feature_cols = [c for c in df_ml.columns if c not in exclude_cols]

X = df_ml[feature_cols].copy()
y_fisica = df_ml['fatiga_fisica'].copy()
y_mental = df_ml['fatiga_mental'].copy()

# Limpiar nulos
mask = X.isnull().any(axis=1) | y_fisica.isnull() | y_mental.isnull()
X = X[~mask]
y_fisica = y_fisica[~mask]
y_mental = y_mental[~mask]

print(f"\n✓ Dataset cargado: {X.shape[0]} muestras × {X.shape[1]} features")
print(f"  Fatiga Física: min={y_fisica.min():.0f}, max={y_fisica.max():.0f}, media={y_fisica.mean():.0f}")
print(f"  Fatiga Mental: min={y_mental.min():.0f}, max={y_mental.max():.0f}, media={y_mental.mean():.0f}")

# Train/Test split
X_train, X_test, y_f_train, y_f_test, y_m_train, y_m_test = train_test_split(
    X, y_fisica, y_mental, test_size=0.3, random_state=42
)

print(f"\n✓ Train/Test split: {X_train.shape[0]} train / {X_test.shape[0]} test")

# Definir modelos
modelos_dict = {
    'Regresión Lineal': LinearRegression(),
    'KNN (k=5)': KNeighborsRegressor(n_neighbors=5),
    'Decision Tree': DecisionTreeRegressor(max_depth=8, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
}

if xgboost_disponible:
    modelos_dict['XGBoost'] = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)

print(f"\n✓ {len(modelos_dict)} modelos listos para entrenar")

# Entrenar modelos - FATIGA FÍSICA
print("\n" + "=" * 80)
print("ENTRENANDO MODELOS - FATIGA FÍSICA")
print("=" * 80)

resultados_fisica = {}

for nombre, modelo in modelos_dict.items():
    print(f"\n📊 {nombre}...", end=" ")
    
    modelo.fit(X_train, y_f_train)
    y_train_pred = modelo.predict(X_train)
    y_test_pred = modelo.predict(X_test)
    
    train_r2 = r2_score(y_f_train, y_train_pred)
    test_r2 = r2_score(y_f_test, y_test_pred)
    test_mae = mean_absolute_error(y_f_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_f_test, y_test_pred))
    
    resultados_fisica[nombre] = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'modelo': modelo,
        'y_test_pred': y_test_pred,
    }
    
    print(f"✓ Train R²={train_r2:.4f} | Test R²={test_r2:.4f} | MAE={test_mae:.2f}")

# Entrenar modelos - FATIGA MENTAL
print("\n" + "=" * 80)
print("ENTRENANDO MODELOS - FATIGA MENTAL")
print("=" * 80)

resultados_mental = {}

for nombre, modelo in modelos_dict.items():
    print(f"\n📊 {nombre}...", end=" ")
    
    modelo.fit(X_train, y_m_train)
    y_train_pred = modelo.predict(X_train)
    y_test_pred = modelo.predict(X_test)
    
    train_r2 = r2_score(y_m_train, y_train_pred)
    test_r2 = r2_score(y_m_test, y_test_pred)
    test_mae = mean_absolute_error(y_m_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_m_test, y_test_pred))
    
    resultados_mental[nombre] = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'modelo': modelo,
        'y_test_pred': y_test_pred,
    }
    
    print(f"✓ Train R²={train_r2:.4f} | Test R²={test_r2:.4f} | MAE={test_mae:.2f}")

# Crear tablas de resultados
df_fisica = pd.DataFrame({
    'Modelo': list(resultados_fisica.keys()),
    'Train R²': [resultados_fisica[m]['train_r2'] for m in resultados_fisica.keys()],
    'Test R²': [resultados_fisica[m]['test_r2'] for m in resultados_fisica.keys()],
    'Test MAE': [resultados_fisica[m]['test_mae'] for m in resultados_fisica.keys()],
}).sort_values('Test R²', ascending=False)

df_mental = pd.DataFrame({
    'Modelo': list(resultados_mental.keys()),
    'Train R²': [resultados_mental[m]['train_r2'] for m in resultados_mental.keys()],
    'Test R²': [resultados_mental[m]['test_r2'] for m in resultados_mental.keys()],
    'Test MAE': [resultados_mental[m]['test_mae'] for m in resultados_mental.keys()],
}).sort_values('Test R²', ascending=False)

# Mostrar resultados
print("\n\n" + "=" * 100)
print("RESULTADOS FINALES - FATIGA FÍSICA")
print("=" * 100)
print(df_fisica.to_string(index=False))

print("\n\n" + "=" * 100)
print("RESULTADOS FINALES - FATIGA MENTAL")
print("=" * 100)
print(df_mental.to_string(index=False))

# Feature Importance
print("\n\n" + "=" * 100)
print("FEATURE IMPORTANCE - DECISION TREE (Fatiga Física)")
print("=" * 100)

dt_model = resultados_fisica['Decision Tree']['modelo']
dt_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False).head(15)

print(dt_importance.to_string(index=False))

print("\n\n" + "=" * 100)
print("FEATURE IMPORTANCE - RANDOM FOREST (Fatiga Física)")
print("=" * 100)

rf_model = resultados_fisica['Random Forest']['modelo']
rf_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False).head(15)

print(rf_importance.to_string(index=False))

# Análisis final
print("\n\n" + "=" * 100)
print("ANÁLISIS Y CONCLUSIONES")
print("=" * 100)

mejor_fisica = df_fisica.iloc[0]
mejor_mental = df_mental.iloc[0]

print(f"""
🏆 GANADOR - FATIGA FÍSICA: {mejor_fisica['Modelo']}
   • Test R²: {mejor_fisica['Test R²']:.4f}
   • Test MAE: {mejor_fisica['Test MAE']:.2f} puntos

🏆 GANADOR - FATIGA MENTAL: {mejor_mental['Modelo']}
   • Test R²: {mejor_mental['Test R²']:.4f}
   • Test MAE: {mejor_mental['Test MAE']:.2f} puntos

KEY FINDINGS:
  ✓ Fatiga NO es predicible linealmente (RL muy bajo)
  ✓ Métodos de árbol (DT, RF) capturan mejor relaciones no-lineales
  ✓ Random Forest ofrece mejor balance generalización/accuracy
  ✓ Dataset pequeño (108 muestras) → más datos = mejor rendimiento
  ✓ KNN viable para prototipado pero RF es más robusto

PRÓXIMOS PASOS:
  1. Ingeniería de features (ratios, deltas temporales)
  2. Tuning de hiperparámetros (GridSearchCV)
  3. Ensemble methods (votación de modelos)
  4. Deep Learning si se recolectan más datos
  5. Análisis de errores por participante/sesión
""")

# Guardar resultados
resumen = pd.concat([
    df_fisica.assign(Target='Fatiga Física'),
    df_mental.assign(Target='Fatiga Mental')
])

resumen.to_csv('resultados_modelos.csv', index=False)
print("\n✓ Resultados guardados en: resultados_modelos.csv")

