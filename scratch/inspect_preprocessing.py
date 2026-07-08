# -*- coding: utf-8 -*-
import json
import sys

# Ensure UTF-8 output encoding
sys.stdout.reconfigure(encoding='utf-8')

notebook_path = "Jupyters/1.Preprocesado/sincronizacion_normalizacion.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print("=== Normalization Code cells in sincronizacion_normalizacion.ipynb ===")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        if "Scaler" in source or "scaler" in source or "normaliz" in source or "scale" in source:
            print(f"\n--- Cell {i+1} ---")
            print(source[:500])
            if len(source) > 500:
                print("...")
