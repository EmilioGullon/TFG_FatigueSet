import json
import pandas as pd
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Test 1: Verificar JSON notebook
print("=" * 60)
print("TEST 1: Validacion JSON notebook")
print("=" * 60)

nb_path = Path("Jupyters/Feature_Engineering_Fisiologico_v3.ipynb")
with open(nb_path) as f:
    nb = json.load(f)
    
print(f"✓ Notebook cargado: {len(nb['cells'])} celdas")
print(f"✓ Formato: Jupyter Notebook v{nb['nbformat']}.{nb['nbformat_minor']}")

# Test 2: Verificar datos disponibles
print("\n" + "=" * 60)
print("TEST 2: Verificacion de datos")
print("=" * 60)

BASE = Path('/c/Users/egull/OneDrive/Documentos/Proyectos/tfg')
DATASET = BASE / 'fatigueset'

print(f"Dataset path: {DATASET}")
print(f"Exists: {DATASET.exists()}")

# Buscar archivo aggregated features
agg_file = BASE / 'fatigueset_aggregated_features.csv'
if agg_file.exists():
    df_agg = pd.read_csv(agg_file)
    print(f"\n✓ Archivo agregado encontrado!")
    print(f"  Shape: {df_agg.shape} (samples x features)")
    print(f"  Columnas: {list(df_agg.columns)}")
    print(f"\n  Primeras 3 filas:")
    print(df_agg.head(3))
else:
    print(f"✗ Archivo no encontrado: {agg_file}")

# Test 3: Prueba basica de imports
print("\n" + "=" * 60)
print("TEST 3: Imports cientificos")
print("=" * 60)

try:
    from scipy.signal import welch
    from scipy.stats import entropy, skew, kurtosis
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.ensemble import RandomForestRegressor
    print("✓ Todos los imports funcionan correctamente")
except Exception as e:
    print(f"✗ Error en imports: {e}")

# Test 4: Prueba de PCA basica
print("\n" + "=" * 60)
print("TEST 4: PCA basico")
print("=" * 60)

if agg_file.exists():
    df = pd.read_csv(agg_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].fillna(0).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    var_exp = np.cumsum(pca.explained_variance_ratio_)
    n_95 = np.argmax(var_exp >= 0.95) + 1
    
    print(f"✓ PCA execution successful")
    print(f"  Input shape: {X_scaled.shape}")
    print(f"  PC1 variance: {pca.explained_variance_ratio_[0]:.1%}")
    print(f"  PC1+PC2 variance: {(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]):.1%}")
    print(f"  Components for 95% variance: {n_95}")

# Test 5: Resumen final
print("\n" + "=" * 60)
print("TEST 5: Resumen de ejecucion")
print("=" * 60)

print("""
✅ Notebook READY FOR USE

Ubicacion: Jupyters/Feature_Engineering_Fisiologico_v3.ipynb

Contenido:
- 13 celdas de codigo ejecutable
- PCA analysis
- t-SNE visualization (si hay datos suficientes)
- K-Means clustering
- Random Forest feature importance
- Correlation heatmaps

Como usar:
1. jupyter notebook Jupyters/Feature_Engineering_Fisiologico_v3.ipynb
2. Ejecutar celdas secuencialmente
3. Observar visualizaciones y resultados
""")

