#!/bin/bash
#SBATCH --job-name=train_dlc
#SBATCH --output=train_dlc_%J.out
#SBATCH --error=train_dlc_%J.err
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --mem=128G

# Load environment
module purge
module load gcc/9.2.0
module load cuda/11.2
module load python/3.9.14
export LD_LIBRARY_PATH=/n/app/python/3.9.14/lib:$LD_LIBRARY_PATH
source /n/scratch/users/a/adm808/envs/DLC_venv_py39/bin/activate

# Run DeepLabCut script
python /home/adm808/Sabatini_Lab/Finalized_Code/train_network.py