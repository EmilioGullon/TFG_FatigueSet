#!/bin/bash
#SBATCH --job-name Foundation_TFG
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

# Temporarily enable SERVER_MODE for both notebooks
sed -i 's/"SERVER_MODE = False"/"SERVER_MODE = True"/g' 09_foundation_models_moment.ipynb
sed -i "s/'SERVER_MODE = False'/'SERVER_MODE = True'/g" 09_foundation_models_moment.ipynb
sed -i 's/SERVER_MODE = False/SERVER_MODE = True/g' 09_foundation_models_moment.ipynb

sed -i 's/"SERVER_MODE = False"/"SERVER_MODE = True"/g' 09_foundation_models_chronos.ipynb
sed -i "s/'SERVER_MODE = False'/'SERVER_MODE = True'/g" 09_foundation_models_chronos.ipynb
sed -i 's/SERVER_MODE = False/SERVER_MODE = True/g' 09_foundation_models_chronos.ipynb

# 1. Run MOMENT fine-tuning in its own python process (Commented out because it is already run successfully)
# echo "[Foundation Server] Starting execution of 09_foundation_models_moment.ipynb..."
# jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3 --inplace 09_foundation_models_moment.ipynb
# echo "[Foundation Server] Finished MOMENT execution!"

# 2. Run Chronos zero-shot in its own fresh python process (guarantees clean GPU memory)
echo "[Foundation Server] Starting execution of 09_foundation_models_chronos.ipynb..."
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3 --inplace 09_foundation_models_chronos.ipynb
echo "[Foundation Server] Finished Chronos execution!"

# Revert back to False for clean Git state
sed -i 's/SERVER_MODE = True/SERVER_MODE = False/g' 09_foundation_models_moment.ipynb
sed -i 's/SERVER_MODE = True/SERVER_MODE = False/g' 09_foundation_models_chronos.ipynb
