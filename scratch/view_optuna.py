# -*- coding: utf-8 -*-
import json

with open('Jupyters/experimento_optuna_optimizadores.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        print(f"=== Cell {i} (Code) ===")
        code = "".join(cell.get('source', []))
        # print first 5 lines and last 5 lines if long
        lines = code.splitlines()
        if len(lines) > 20:
            print("\n".join(lines[:10]))
            print("...")
            print("\n".join(lines[-10:]))
        else:
            print(code)
        print("======================\n")
