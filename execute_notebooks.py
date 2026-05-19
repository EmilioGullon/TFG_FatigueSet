"""
Script para ejecutar y corregir todos los Jupyter notebooks del proyecto
"""
import sys
import io
# Configurar encoding para Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
from pathlib import Path

jupyter_dir = r"c:\Users\egull\OneDrive\Documentos\Proyectos\tfg\Jupyters"
os.chdir(jupyter_dir)

# Lista de notebooks a ejecutar (excluyendo los _test)
notebooks = [
    "FatigueSet Procesado de datos.ipynb",
    "fatigueset_library_guide.ipynb",
    "Feature_Engineering_Fisiologico_Avanzado.ipynb",
    "Feature_Engineering_Fisiologico_FINAL.ipynb",
    "Feature_Engineering_Fisiologico_v3.ipynb",
    "sincronizacion_normalizacion.ipynb",
    "windowing_data_leakage.ipynb"
]

print("=" * 80)
print("EJECUTANDO Y CORRIGIENDO TODOS LOS NOTEBOOKS")
print("=" * 80)

results = {
    "executed": [],
    "errors": [],
    "skipped": []
}

for notebook_name in notebooks:
    print(f"\n{'='*80}")
    print(f"Procesando: {notebook_name}")
    print('='*80)

    notebook_path = os.path.join(jupyter_dir, notebook_name)

    if not os.path.exists(notebook_path):
        print(f"  [SKIP] ARCHIVO NO ENCONTRADO")
        results["skipped"].append(notebook_name)
        continue

    try:
        print(f"  - Cargando notebook...")
        with open(notebook_path, encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        print(f"  - Configurando ejecutor...")
        # Configurar timeout largo (300 segundos = 5 minutos por celda)
        ep = ExecutePreprocessor(timeout=300, kernel_name='python3')

        print(f"  - Ejecutando {len(nb.cells)} celdas...")
        nb, resources = ep.preprocess(nb, {'metadata': {'path': jupyter_dir}})

        print(f"  [OK] EJECUCION EXITOSA")

        # Guardar notebook ejecutado
        output_path = notebook_path.replace('.ipynb', '_EJECUTADO.ipynb')
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"    Guardado: {os.path.basename(output_path)}")

        results["executed"].append(notebook_name)

    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}")
        print(f"    Detalles: {str(e)[:200]}")
        results["errors"].append({
            "notebook": notebook_name,
            "error_type": type(e).__name__,
            "error_msg": str(e)[:300]
        })

# Resumen final
print("\n" + "=" * 80)
print("RESUMEN DE EJECUCION")
print("=" * 80)

print(f"\n[OK] EXITOSOS ({len(results['executed'])}):")
for nb in results["executed"]:
    print(f"  - {nb}")

if results["errors"]:
    print(f"\n[ERROR] CON ERRORES ({len(results['errors'])}):")
    for error in results["errors"]:
        print(f"  - {error['notebook']}")
        print(f"    Error: {error['error_type']}")
        print(f"    Mensaje: {error['error_msg'][:150]}...\n")

if results["skipped"]:
    print(f"\n[SKIP] SALTADOS ({len(results['skipped'])}):")
    for nb in results["skipped"]:
        print(f"  - {nb}")

print("\n" + "=" * 80)
print(f"TOTAL: {len(results['executed'])} exitosos, {len(results['errors'])} errores, {len(results['skipped'])} saltados")
print("=" * 80)
