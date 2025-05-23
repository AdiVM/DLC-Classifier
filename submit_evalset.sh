#!/bin/bash
#SBATCH --job-name=train_set_overlay
#SBATCH --output=trainset_job_%J.out
#SBATCH --error=trainset_job_%J.err
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --partition=short
#SBATCH --mem=128G


source /n/groups/patel/adithya/scenv/bin/activate

cd /home/adm808/Sabatini_Lab/Finalized_Code

# Run frame processing script
python3 eval_set_create.py