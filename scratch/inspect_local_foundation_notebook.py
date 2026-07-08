"""Parses the local executed notebook 09_foundation_models.ipynb and prints cell outputs."""
import json
import sys
from pathlib import Path

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    try:
        nb_path = Path("Jupyters/09_foundation_models.ipynb")
        if not nb_path.exists():
            print(f"[ERROR] Notebook {nb_path} does not exist locally.")
            sys.exit(1)
            
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
            
        print("=== EXECUTED CELL OUTPUTS IN 09_FOUNDATION_MODELS.IPYNB ===")
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source_lines = cell.get('source', [])
                source_preview = "".join(source_lines[:2]).strip().replace('\n', ' ')
                outputs = cell.get('outputs', [])
                if outputs:
                    print(f"\n--- Cell {i} (Preview: {source_preview}...) ---")
                    for out in outputs:
                        if out.get('output_type') == 'stream':
                            print("".join(out.get('text', [])).strip())
                        elif out.get('output_type') == 'error':
                            print("  [ERROR]:")
                            print("".join(out.get('traceback', [])))
                        elif out.get('output_type') == 'execute_result':
                            print("  [EXECUTE RESULT]:")
                            print("".join(out.get('data', {}).get('text/plain', [])))
    except Exception as e:
        print(f"[ERROR] Parsing local notebook failed: {e}")

if __name__ == "__main__":
    main()
