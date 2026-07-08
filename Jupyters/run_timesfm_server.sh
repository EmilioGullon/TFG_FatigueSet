#!/bin/bash
#SBATCH --job-name TimesFM_TFG
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

# Temporarily enable SERVER_MODE
sed -i 's/"SERVER_MODE = False"/"SERVER_MODE = True"/g' 10_timesfm.ipynb
sed -i "s/'SERVER_MODE = False'/'SERVER_MODE = True'/g" 10_timesfm.ipynb
sed -i 's/SERVER_MODE = False/SERVER_MODE = True/g' 10_timesfm.ipynb

echo "[TimesFM Server] Starting execution of 10_timesfm.ipynb..."
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3 --inplace 10_timesfm.ipynb
echo "[TimesFM Server] Finished execution!"

# Revert back to False
sed -i 's/SERVER_MODE = True/SERVER_MODE = False/g' 10_timesfm.ipynb
