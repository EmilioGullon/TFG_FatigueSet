"""Dumps all Jupyter notebook outputs of TimesFM notebook to a text file."""
import json

try:
    with open('Jupyters/10_timesfm.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    with open('scratch/timesfm_outputs.txt', 'w', encoding='utf-8') as out_f:
        for idx, cell in enumerate(nb['cells']):
            ct = cell['cell_type']
            out_f.write(f"\n=========================================================\n")
            out_f.write(f"CELL {idx+1} ({ct})\n")
            out_f.write(f"=========================================================\n")
            
            # Source
            source = "".join(cell['source'])
            out_f.write(f"--- SOURCE ---\n{source}\n\n")
            
            # Outputs
            if ct == 'code':
                outputs = cell.get('outputs', [])
                out_f.write(f"--- OUTPUTS ({len(outputs)}) ---\n")
                for out in outputs:
                    if 'text' in out:
                        text_val = out['text']
                        if isinstance(text_val, list):
                            out_f.write("".join(text_val))
                        else:
                            out_f.write(str(text_val))
                    elif 'data' in out:
                        data = out['data']
                        if 'text/plain' in data:
                            plain_val = data['text/plain']
                            if isinstance(plain_val, list):
                                out_f.write("".join(plain_val) + "\n")
                            else:
                                out_f.write(str(plain_val) + "\n")
                    elif 'traceback' in out:
                        trace_val = out['traceback']
                        if isinstance(trace_val, list):
                            out_f.write("\n".join(trace_val) + "\n")
                        else:
                            out_f.write(str(trace_val) + "\n")
    print("[OK] Notebook outputs dumped to scratch/timesfm_outputs.txt")
except Exception as e:
    print(f"[ERROR] Failed to dump notebook: {e}")
