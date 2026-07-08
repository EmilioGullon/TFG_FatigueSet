# -*- coding: utf-8 -*-
import json
import sys

# Ensure UTF-8 output encoding
sys.stdout.reconfigure(encoding='utf-8')

notebook_path = "Jupyters/09_foundation_models.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print("=== Cell Outputs of 09_foundation_models.ipynb ===")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))[:100].replace('\n', ' ')
        print(f"\n--- Cell {i} (Code): {source} ---")
        outputs = cell.get('outputs', [])
        print(f"Total outputs: {len(outputs)}")
        for j, out in enumerate(outputs):
            ot = out.get('output_type')
            print(f"  Output {j} type: {ot}")
            if ot == 'stream':
                text = "".join(out.get('text', []))
                print("  [STREAM]")
                print("\n".join(text.splitlines()[:15]))
                if len(text.splitlines()) > 15:
                    print("  ...")
            elif ot == 'execute_result':
                data = out.get('data', {}).get('text/plain', [])
                print("  [EXECUTE RESULT]")
                print("".join(data)[:300])
            elif ot == 'error':
                ename = out.get('ename')
                evalue = out.get('evalue')
                traceback = "\n".join(out.get('traceback', []))
                print(f"  [ERROR] {ename}: {evalue}")
                print(traceback[:1000])
