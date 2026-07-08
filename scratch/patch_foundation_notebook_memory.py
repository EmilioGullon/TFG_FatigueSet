"""Patches Jupyters/09_foundation_models.ipynb to add GPU memory cleanup after MOMENT execution."""
import json
from pathlib import Path

notebook_path = Path("Jupyters/09_foundation_models.ipynb")

def main():
    if not notebook_path.exists():
        print(f"[ERROR] Notebook not found at: {notebook_path}")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    patched_count = 0

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue

        source = cell['source']
        source_str = "".join(source)

        # Find the MOMENT fine-tuning execution cell
        if "finetune_moment_kfold(" in source_str:
            print(f"Patching MOMENT execution cell {i}...")
            
            # Find the end of the cell or append to it
            target = '    print(f"\\nTiempo total fine-tuning: {MOMENT_TIME:.1f}s")\nelse:\n    print("[SKIP] MOMENT no disponible — instala momentfm e intenta de nuevo.")'
            # Let's search for the standard else case as well
            target_simple = '    print(f"\\nTiempo total fine-tuning: {MOMENT_TIME:.1f}s")'
            
            # We want to append cleanup lines at the end of the source string
            cleanup_lines = (
                "\n\n# Limpiar memoria GPU explícitamente para dar espacio a Chronos\n"
                "import gc\n"
                "gc.collect()\n"
                "if torch.cuda.is_available():\n"
                "    torch.cuda.empty_cache()\n"
            )
            
            # Simply append to the end of the cell if it's the right cell
            source_str = source_str.strip() + cleanup_lines
            cell['source'] = [line + "\n" for line in source_str.split("\n")]
            patched_count += 1
            break

    # Save the patched notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"[OK] Notebook patching completed. Patched cells count: {patched_count}")

if __name__ == "__main__":
    main()
