# -*- coding: utf-8 -*-
import json

notebook_path = "Jupyters/experimento_optuna_optimizadores.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

modified = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source_text = "".join(cell["source"])
        if "N_TRIALS = 50" in source_text:
            print("Found cell with N_TRIALS = 50. Modifying...")
            # We replace:
            # N_TRIALS = 50 -> N_TRIALS = 20
            # N_CV_SPLITS = 5 -> N_CV_SPLITS = 3
            # EPOCHS_PER_TRIAL = 20 -> EPOCHS_PER_TRIAL = 10
            new_source = []
            for line in cell["source"]:
                if "N_TRIALS = 50" in line:
                    line = line.replace("N_TRIALS = 50", "N_TRIALS = 20")
                elif "N_CV_SPLITS = 5" in line:
                    line = line.replace("N_CV_SPLITS = 5", "N_CV_SPLITS = 3")
                elif "EPOCHS_PER_TRIAL = 20" in line:
                    line = line.replace("EPOCHS_PER_TRIAL = 20", "EPOCHS_PER_TRIAL = 10")
                new_source.append(line)
            cell["source"] = new_source
            modified = True

        if "FINAL_EPOCHS = 50" in source_text:
            print("Found cell with FINAL_EPOCHS = 50. Modifying...")
            # We replace:
            # FINAL_EPOCHS = 50 -> FINAL_EPOCHS = 30
            # PATIENCE = 10 -> PATIENCE = 8
            new_source = []
            for line in cell["source"]:
                if "FINAL_EPOCHS = 50" in line:
                    line = line.replace("FINAL_EPOCHS = 50", "FINAL_EPOCHS = 30")
                elif "PATIENCE = 10" in line:
                    line = line.replace("PATIENCE = 10", "PATIENCE = 8")
                new_source.append(line)
            cell["source"] = new_source
            modified = True

if modified:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook modified successfully.")
else:
    print("No cells matched pattern to modify.")
