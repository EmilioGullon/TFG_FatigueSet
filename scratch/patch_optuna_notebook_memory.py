"""Patches Jupyters/experimento_optuna_optimizadores.ipynb to add GPU memory cleanup and respect SERVER_MODE."""
import json
import os
from pathlib import Path

notebook_path = Path("Jupyters/experimento_optuna_optimizadores.ipynb")

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

        # 1. Patch quick_cv_eval to add memory cleanup
        if "def quick_cv_eval(" in source_str and "trial.should_prune():" in source_str:
            print(f"Patching quick_cv_eval in cell {i}...")
            
            # Find the pruning check block and replace it with memory cleanup
            target = (
                "        # Pruning de Optuna: descartar trial si el fold actual no muestra potencial\n"
                "        if trial is not None:\n"
                "            trial.report(np.mean(val_losses), step=fold_idx)\n"
                "            if trial.should_prune():\n"
                "                raise optuna.exceptions.TrialPruned()\n"
            )
            
            replacement = (
                "        # Pruning de Optuna: descartar trial si el fold actual no muestra potencial\n"
                "        prune_trial = False\n"
                "        if trial is not None:\n"
                "            trial.report(np.mean(val_losses), step=fold_idx)\n"
                "            if trial.should_prune():\n"
                "                prune_trial = True\n"
                "\n"
                "        # Limpiar memoria GPU explícitamente para evitar CUDA OOM\n"
                "        del model, optimizer, train_loader, val_loader\n"
                "        import gc\n"
                "        gc.collect()\n"
                "        if torch.cuda.is_available():\n"
                "            torch.cuda.empty_cache()\n"
                "\n"
                "        if prune_trial:\n"
                "            raise optuna.exceptions.TrialPruned()\n"
            )
            
            if target in source_str:
                source_str = source_str.replace(target, replacement)
                cell['source'] = [line + "\n" if not line.endswith("\n") else line for line in source_str.split("\n")]
                # remove the trailing empty string if split leaves one
                if cell['source'] and cell['source'][-1] == "\n":
                    cell['source'].pop()
                patched_count += 1
            else:
                # Try a looser match if exact indent differs
                print("Exact target not found, trying fuzzy match...")
                if "raise optuna.exceptions.TrialPruned()" in source_str:
                    # Let's replace the whole block manually
                    # Just replace the last part of the loop before returning
                    idx = source_str.find("if trial is not None:")
                    if idx != -1:
                        loop_end = source_str[idx:]
                        # replace the pruning block in this segment
                        new_loop_end = (
                            "prune_trial = False\n"
                            "        if trial is not None:\n"
                            "            trial.report(np.mean(val_losses), step=fold_idx)\n"
                            "            if trial.should_prune():\n"
                            "                prune_trial = True\n\n"
                            "        del model, optimizer, train_loader, val_loader\n"
                            "        import gc\n"
                            "        gc.collect()\n"
                            "        if torch.cuda.is_available():\n"
                            "            torch.cuda.empty_cache()\n\n"
                            "        if prune_trial:\n"
                            "            raise optuna.exceptions.TrialPruned()\n"
                            "\n"
                            "    return float(np.mean(val_losses))"
                        )
                        source_str = source_str[:idx] + new_loop_end
                        cell['source'] = [line + "\n" for line in source_str.split("\n")]
                        patched_count += 1

        # 2. Patch hardcoded N_TRIALS to respect SERVER_MODE
        elif "N_TRIALS = 15" in source_str and "for model_name, sampler_fn in MODEL_SAMPLERS.items():" in source_str:
            print(f"Patching N_TRIALS to respect SERVER_MODE in cell {i}...")
            target = (
                "# Configuraci\u00f3n global del experimento Optuna\n"
                "N_TRIALS = 15        # N\u00famero de ensayos bayesianos por modelo\n"
                "EPOCHS_PER_TRIAL = 5 # \u00c9pocas de entrenamiento por ensayo (sanity check r\u00e1pido)\n"
                "N_CV_SPLITS = 3      # Folds de validaci\u00f3n cruzada\n"
            )
            # Try both raw unicode and standard chars
            target_ascii = (
                "# Configuración global del experimento Optuna\n"
                "N_TRIALS = 15        # Número de ensayos bayesianos por modelo\n"
                "EPOCHS_PER_TRIAL = 5 # Épocas de entrenamiento por ensayo (sanity check rápido)\n"
                "N_CV_SPLITS = 3      # Folds de validación cruzada\n"
            )
            
            replacement = (
                "# Configuración global del experimento Optuna\n"
                "if 'SERVER_MODE' in globals() and SERVER_MODE:\n"
                "    N_TRIALS = 50        # Más ensayos bayesianos en servidor\n"
                "    N_CV_SPLITS = 5      # 5-fold CV para rigor científico\n"
                "    EPOCHS_PER_TRIAL = 20# Más épocas por trial\n"
                "else:\n"
                "    N_TRIALS = 15\n"
                "    N_CV_SPLITS = 3\n"
                "    EPOCHS_PER_TRIAL = 5\n"
            )
            
            matched = False
            for t in [target, target_ascii]:
                if t in source_str:
                    source_str = source_str.replace(t, replacement)
                    matched = True
                    break
            
            if not matched:
                # Let's try simple replacement for the variables
                source_str = source_str.replace("N_TRIALS = 15", "N_TRIALS = 50 if ('SERVER_MODE' in globals() and SERVER_MODE) else 15")
                source_str = source_str.replace("EPOCHS_PER_TRIAL = 5", "EPOCHS_PER_TRIAL = 20 if ('SERVER_MODE' in globals() and SERVER_MODE) else 5")
                source_str = source_str.replace("N_CV_SPLITS = 3", "N_CV_SPLITS = 5 if ('SERVER_MODE' in globals() and SERVER_MODE) else 3")
                matched = True
                
            if matched:
                cell['source'] = [line + "\n" if not line.endswith("\n") else line for line in source_str.split("\n")]
                patched_count += 1

        # 3. Patch final evaluation fold loop to clean memory
        elif "for train_idx, val_idx in kf_final.split(" in source_str and "fold_r2.append(r2_score(t_arr, p_arr))" in source_str:
            print(f"Patching final evaluation fold loop in cell {i}...")
            target = "        fold_r2.append(r2_score(t_arr, p_arr))\n"
            replacement = (
                "        fold_r2.append(r2_score(t_arr, p_arr))\n"
                "\n"
                "        # Limpiar memoria GPU explícitamente\n"
                "        del model, optimizer, train_loader, val_loader\n"
                "        import gc\n"
                "        gc.collect()\n"
                "        if torch.cuda.is_available():\n"
                "            torch.cuda.empty_cache()\n"
            )
            if target in source_str:
                source_str = source_str.replace(target, replacement)
                cell['source'] = [line + "\n" if not line.endswith("\n") else line for line in source_str.split("\n")]
                patched_count += 1

        # 4. Patch baseline fold loop to clean memory
        elif "for train_idx, val_idx in kf_base.split(" in source_str and "fold_r2.append(r2_score(t_arr, p_arr))" in source_str:
            print(f"Patching baseline fold loop in cell {i}...")
            target = "        fold_r2.append(r2_score(t_arr, p_arr))\n"
            replacement = (
                "        fold_r2.append(r2_score(t_arr, p_arr))\n"
                "\n"
                "        # Limpiar memoria GPU explícitamente\n"
                "        del model, optimizer, train_loader, val_loader\n"
                "        import gc\n"
                "        gc.collect()\n"
                "        if torch.cuda.is_available():\n"
                "            torch.cuda.empty_cache()\n"
            )
            if target in source_str:
                source_str = source_str.replace(target, replacement)
                cell['source'] = [line + "\n" if not line.endswith("\n") else line for line in source_str.split("\n")]
                patched_count += 1

    # Save the patched notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"[OK] Notebook patching completed. Patched cells count: {patched_count}")

if __name__ == "__main__":
    main()
