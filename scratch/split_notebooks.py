# -*- coding: utf-8 -*-
import json
import os

source_nb_path = "Jupyters/09_foundation_models.ipynb"
moment_nb_path = "Jupyters/09_foundation_models_moment.ipynb"
chronos_nb_path = "Jupyters/09_foundation_models_chronos.ipynb"

with open(source_nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# -------------------------------------------------------------
# 1. Create MOMENT Notebook
# -------------------------------------------------------------
# We force Chronos to be disabled
moment_nb = json.loads(json.dumps(nb))  # Deep copy
for cell in moment_nb["cells"]:
    if cell["cell_type"] == "code":
        source_text = "".join(cell["source"])
        if "SANITY CHECK: Chronos zero-shot" in source_text:
            # Inject CHRONOS_AVAILABLE = False at the top of the cell
            print("Disabling Chronos in MOMENT notebook...")
            cell["source"] = [
                "# Force disabled to execute MOMENT in separate process\n",
                "CHRONOS_AVAILABLE = False\n",
                "print('[INFO] Chronos disabled for separate MOMENT run.')\n"
            ]

with open(moment_nb_path, "w", encoding="utf-8") as f:
    json.dump(moment_nb, f, indent=1, ensure_ascii=False)
print(f"Created: {moment_nb_path}")

# -------------------------------------------------------------
# 2. Create Chronos Notebook
# -------------------------------------------------------------
# We force MOMENT to be disabled, and we modify the final cell to load MOMENT results if available
chronos_nb = json.loads(json.dumps(nb))  # Deep copy
for cell in chronos_nb["cells"]:
    if cell["cell_type"] == "code":
        source_text = "".join(cell["source"])
        if "SANITY CHECK: MOMENT forward pass" in source_text:
            # Inject MOMENT_AVAILABLE = False
            print("Disabling MOMENT in Chronos notebook...")
            cell["source"] = [
                "# Force disabled to execute Chronos in separate process\n",
                "MOMENT_AVAILABLE = False\n",
                "print('[INFO] MOMENT disabled for separate Chronos run.')\n"
            ]
        elif "Construir tabla unificada con MOMENT y Chronos" in source_text:
            print("Updating comparison table generation in Chronos notebook...")
            # We rewrite this cell to load previous results first
            new_source = [
                "def mean_metric(results_list, key):\n",
                "    \"\"\"Media de una métrica sobre todos los folds.\"\"\"\n",
                "    vals = [r[key] for r in results_list if r.get(key) is not None]\n",
                "    return np.mean(vals) if vals else float('nan')\n",
                "\n",
                "# Cargar resultados anteriores de MOMENT si existen para combinarlos\n",
                "import os\n",
                "output_dir = Path.cwd().parent / \"output\"\n",
                "csv_path = output_dir / \"resultados_foundation_models.csv\"\n",
                "rows = []\n",
                "if os.path.exists(csv_path):\n",
                "    try:\n",
                "        df_old = pd.read_csv(csv_path)\n",
                "        # Conservar solo la fila de MOMENT\n",
                "        df_old = df_old[df_old[\"Modelo\"].str.contains(\"MOMENT\", na=False)]\n",
                "        rows = df_old.to_dict('records')\n",
                "        print(f'[INFO] Cargados resultados anteriores de MOMENT ({len(rows)} filas).')\n",
                "    except Exception as e:\n",
                "        print(f'[Warning] Error al cargar resultados anteriores: {e}')\n",
                "\n",
                "if CHRONOS_RESULTS:\n",
                "    rows.append({\n",
                "        \"Modelo\": f\"Chronos ({CHRONOS_CKPT.split('/')[-1]})\",\n",
                "        \"Tipo\": \"Foundation (decoder probabilístico)\",\n",
                "        \"Paradigma\": \"Zero-shot + Linear probe\",\n",
                "        \"Probabilístico\": \"Sí\",\n",
                "        \"MAE Física\": mean_metric(CHRONOS_RESULTS, 'mae_fisica'),\n",
                "        \"RMSE Física\": mean_metric(CHRONOS_RESULTS, 'rmse_fisica'),\n",
                "        \"R² Física\": mean_metric(CHRONOS_RESULTS, 'r2_fisica'),\n",
                "        \"MAE Mental\": mean_metric(CHRONOS_RESULTS, 'mae_mental'),\n",
                "        \"RMSE Mental\": mean_metric(CHRONOS_RESULTS, 'rmse_mental'),\n",
                "        \"R² Mental\": mean_metric(CHRONOS_RESULTS, 'r2_mental'),\n",
                "        \"CRPS Física\": PROB_METRICS.get('crps_fisica_mean', float('nan')),\n",
                "        \"CRPS Mental\": PROB_METRICS.get('crps_mental_mean', float('nan')),\n",
                "        \"Cobertura 90%\": PROB_METRICS.get('coverage90_fisica', float('nan')),\n",
                "        \"Nº Params (entrenables)\": float('nan'),\n",
                "        \"Tiempo (s)\": CHRONOS_TIME,\n",
                "    })\n",
                "\n",
                "if rows:\n",
                "    df_comparativa = pd.DataFrame(rows)\n",
                "    print(\"\\n=== TABLA COMPARATIVA: Modelos Fundacionales ===\")\n",
                "    print(df_comparativa.to_string(index=False))\n",
                "else:\n",
                "    df_comparativa = pd.DataFrame()\n"
            ]
            cell["source"] = new_source

with open(chronos_nb_path, "w", encoding="utf-8") as f:
    json.dump(chronos_nb, f, indent=1, ensure_ascii=False)
print(f"Created: {chronos_nb_path}")
