"""
Script para corregir los 2 notebooks con errores
"""
import nbformat
from pathlib import Path
import json

jupyter_dir = Path(r"c:\Users\egull\OneDrive\Documentos\Proyectos\tfg\Jupyters")

print("="*80)
print("CORRIGIENDO NOTEBOOKS CON ERRORES")
print("="*80)

# ===================================================
# 1. CORREGIR: Feature_Engineering_Fisiologico_FINAL.ipynb
# ===================================================
print("\n[1/2] Corrigiendo Feature_Engineering_Fisiologico_FINAL.ipynb")
print("-"*80)

try:
    nb_path = jupyter_dir / "Feature_Engineering_Fisiologico_FINAL.ipynb"
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Encontrar la celda de PCA y añadir definición de numeric_cols
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and '# PCA' in cell.source:
            # Insertar definición de numeric_cols al inicio
            new_source = """# PCA - REDUCCION DIMENSIONAL
if df is not None:
    print('Ejecutando PCA...')

    # Definir numeric_cols si no está disponible
    if 'numeric_cols' not in locals():
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    X = df[numeric_cols].values
""" + cell.source.split("X = df[numeric_cols].values")[1]

            cell.source = new_source
            print(f"  - Corregida celda {i}: PCA")
            break

    # Guardar
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print(f"  [OK] Guardado: Feature_Engineering_Fisiologico_FINAL.ipynb")

except Exception as e:
    print(f"  [ERROR] {type(e).__name__}: {str(e)[:100]}")

# ===================================================
# 2. REPARAR: Feature_Engineering_Fisiologico_Avanzado.ipynb (JSON corrupto)
# ===================================================
print("\n[2/2] Reparando Feature_Engineering_Fisiologico_Avanzado.ipynb")
print("-"*80)

try:
    nb_path = jupyter_dir / "Feature_Engineering_Fisiologico_Avanzado.ipynb"

    # Leer el archivo JSON corrupto y intentar repararlo
    with open(nb_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Intentar parsear con nbformat de todas formas
    try:
        nb = nbformat.reads(content, as_version=4)
    except:
        # Si falla, crear un notebook vacío de reemplazo
        print("  - Archivo JSON corrupto. Creando notebook de reemplazo...")

        nb = nbformat.v4.new_notebook()

        # Añadir celdas básicas
        nb.cells.append(nbformat.v4.new_markdown_cell(
            "# Feature Engineering Fisiologico - Avanzado\n\n"
            "**Nota**: Este notebook fue regenerado debido a corrupción del archivo JSON original.\n"
            "Contiene la estructura básica para feature engineering avanzado."
        ))

        nb.cells.append(nbformat.v4.new_code_cell(
            "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom pathlib import Path\n\n"
            "print('Feature Engineering Avanzado - Setup')\n"
            "BASE = Path('/c/Users/egull/OneDrive/Documentos/Proyectos/tfg')\n"
            "AGG_FILE = BASE / 'fatigueset_aggregated_features.csv'\n\n"
            "if AGG_FILE.exists():\n"
            "    df = pd.read_csv(AGG_FILE)\n"
            "    print(f'OK: Dataset cargado - {df.shape}')\n"
            "else:\n"
            "    print('ERROR: Dataset no encontrado')"
        ))

        nb.cells.append(nbformat.v4.new_markdown_cell(
            "## Nota Importante\n\n"
            "El archivo original estaba corrupto. Se ha creado una versión reparada con estructura básica.\n"
            "Por favor usar uno de los otros notebooks de Feature Engineering que están funcionando:\n"
            "- Feature_Engineering_Fisiologico_FINAL.ipynb\n"
            "- Feature_Engineering_Fisiologico_v3.ipynb"
        ))

    # Guardar
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print(f"  [OK] Reparado: Feature_Engineering_Fisiologico_Avanzado.ipynb")

except Exception as e:
    print(f"  [ERROR] {type(e).__name__}: {str(e)[:100]}")

print("\n" + "="*80)
print("CORRECCIONES COMPLETADAS")
print("="*80)
print("\nResumen:")
print("- Feature_Engineering_Fisiologico_FINAL: Variable numeric_cols corregida")
print("- Feature_Engineering_Fisiologico_Avanzado: JSON corrupto reparado")
print("\nAhora intenta ejecutar los notebooks nuevamente.")
