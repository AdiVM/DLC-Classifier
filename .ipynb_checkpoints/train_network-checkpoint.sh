#!/bin/bash
#SBATCH --job-name=train_dlc
#SBATCH --output=train_dlc_%J.out
#SBATCH --error=train_dlc_%J.err
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --partition=gpu                # <<< Use GPU partition
#SBATCH --gres=gpu:1                   # <<< Request 1 GPU
#SBATCH --mem=64G                      
#SBATCH -c 4                       

# Load environment
module purge
module load gcc/9.2.0
module load cuda/11.2
module load python/3.9.14
# Check if this is needed later
export LD_LIBRARY_PATH=/n/app/python/3.9.14/lib:$LD_LIBRARY_PATH

# Activate your GPU-enabled DeepLabCut environment
source /n/scratch/users/a/adm808/envs/DLC_venv_py39/bin/activate

# Confirm GPU is visible
echo "GPUs available:"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Run DLC Script
python /home/adm808/Sabatini_Lab/Finalized_Code/train_network.py