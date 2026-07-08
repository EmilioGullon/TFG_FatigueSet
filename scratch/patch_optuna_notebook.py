"""Patches the Optuna notebook to include SERVER_MODE configuration switcher."""
import json

path = 'Jupyters/experimento_optuna_optimizadores.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find cell with SEED = 42 and prepend SERVER_MODE = False
patched_seed = False
for cell in nb['cells']:
    source = cell.get('source', [])
    source_str = "".join(source)
    if 'SEED = 42' in source_str and 'SERVER_MODE' not in source_str:
        # Insert SERVER_MODE at the beginning
        new_source = [
            "# CONFIGURACIÓN GLOBAL: False = pruebas locales | True = servidor potente\n",
            "SERVER_MODE = False\n",
            "\n"
        ] + source
        cell['source'] = new_source
        patched_seed = True
        print("[OK] Patched cell 3 (SEED and SERVER_MODE)")
        break

# Find cell with N_TRIALS = 15 and change it to server/local conditional
patched_trials = False
for cell in nb['cells']:
    source = cell.get('source', [])
    source_str = "".join(source)
    if 'N_TRIALS = 15' in source_str:
        new_source = [
            "# Configuración global del experimento Optuna\n",
            "if 'SERVER_MODE' in globals() and SERVER_MODE:\n",
            "    N_TRIALS = 50        # Más ensayos bayesianos en servidor\n",
            "    N_CV_SPLITS = 5      # 5-fold CV para rigor científico\n",
            "    EPOCHS_PER_TRIAL = 20# Más épocas por trial\n",
            "else:\n",
            "    N_TRIALS = 15        # Ensayos locales para pruebas rápidas\n",
            "    N_CV_SPLITS = 3\n",
            "    EPOCHS_PER_TRIAL = 5\n",
            "\n",
            "# Definición de las familias de modelos con sus funciones de muestreo\n",
            "MODEL_SAMPLERS = {\n",
            "    \"Custom LSTM\":        sample_lstm,\n",
            "    \"Custom GRU\":         sample_gru,\n",
            "    \"Custom CNN-LSTM\":    sample_cnn_lstm,\n",
            "    \"Custom TCN\":         sample_tcn,\n",
            "    \"Custom Transformer\": sample_transformer,\n",
            "    \"Custom PatchTST\":    sample_patchtst,\n",
            "    \"Custom xLSTM\":       sample_xlstm,\n",
            "}\n",
            "\n",
            "# Almacenamiento de los mejores hiperparámetros por modelo\n",
            "best_hyperparams_per_model = {}\n",
            "optuna_results = []\n"
        ]
        cell['source'] = new_source
        patched_trials = True
        print("[OK] Patched cell 12 (N_TRIALS conditional)")
        break

# Find cell with FINAL_EPOCHS = 15 and condition it
patched_epochs = False
for cell in nb['cells']:
    source = cell.get('source', [])
    source_str = "".join(source)
    if 'FINAL_EPOCHS = 15' in source_str:
        new_source = []
        for line in source:
            if 'FINAL_EPOCHS = 15' in line:
                new_source.append("if 'SERVER_MODE' in globals() and SERVER_MODE:\n")
                new_source.append("    FINAL_EPOCHS = 50\n")
                new_source.append("    PATIENCE = 10\n")
                new_source.append("else:\n")
                new_source.append("    FINAL_EPOCHS = 15\n")
                new_source.append("    PATIENCE = 5\n")
            elif 'PATIENCE = 5' in line:
                # Handled in conditional above
                pass
            else:
                new_source.append(line)
        cell['source'] = new_source
        patched_epochs = True
        print("[OK] Patched cell 18 (FINAL_EPOCHS conditional)")
        break

if patched_seed and patched_trials and patched_epochs:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("[SUCCESS] Optuna notebook patched successfully.")
else:
    print("[ERROR] Failed to patch all required cells.")
