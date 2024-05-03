#!/bin/bash

#SBATCH --job-name=AE_RE_allfolds_20epochs_1layer
#SBATCH --partition=GPUp4
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH -o job_%j.out
#SBATCH -e job_%j.err
export CUDA_VISIBLE_DEVICES=0
module load cuda118
module load cuda118/toolkit/11.8.0
module load parallel
module load python/3.7.x-anaconda
source activate /archive/bioinformatics/DLLab/shared/CondaEnvironments/Aixa_ARMED_2

python run_AE_RE_allfolds.py
