#!/bin/bash
#SBATCH --job-name Optuna_TFG
#SBATCH --partition dios
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/homeGPU/egullonl01/tfg/Jupyters/slurm-%j.out
#SBATCH --error=/mnt/homeGPU/egullonl01/tfg/Jupyters/slurm-%j.out

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"

# Redirect cache and config directories to homeGPU to prevent home quota issues
export IPYTHONDIR="/mnt/homeGPU/egullonl01/.ipython"
export JUPYTER_CONFIG_DIR="/mnt/homeGPU/egullonl01/.jupyter"
export JUPYTER_DATA_DIR="/mnt/homeGPU/egullonl01/.local/share/jupyter"
export JUPYTER_RUNTIME_DIR="/mnt/homeGPU/egullonl01/.local/share/jupyter/runtime"
export HF_HOME="/mnt/homeGPU/egullonl01/.cache/huggingface"
export XDG_CACHE_HOME="/mnt/homeGPU/egullonl01/.cache"

eval "$(conda shell.bash hook)"

conda activate /mnt/homeGPU/egullonl01/conda_tfg
cd /mnt/homeGPU/egullonl01/tfg/Jupyters

# Temporarily enable SERVER_MODE for server execution scale
sed -i 's/"SERVER_MODE = False"/"SERVER_MODE = True"/g' experimento_optuna_optimizadores.ipynb
sed -i "s/'SERVER_MODE = False'/'SERVER_MODE = True'/g" experimento_optuna_optimizadores.ipynb
sed -i 's/SERVER_MODE = False/SERVER_MODE = True/g' experimento_optuna_optimizadores.ipynb

echo "[Optuna Server] Starting execution of experimento_optuna_optimizadores.ipynb..."
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3 --inplace experimento_optuna_optimizadores.ipynb
echo "[Optuna Server] Finished execution!"

# Revert back to False for clean Git state
sed -i 's/SERVER_MODE = True/SERVER_MODE = False/g' experimento_optuna_optimizadores.ipynb
