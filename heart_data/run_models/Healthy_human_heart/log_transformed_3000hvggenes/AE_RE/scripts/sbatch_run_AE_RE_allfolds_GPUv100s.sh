#!/bin/bash

#SBATCH --job-name=AE_RE_allfolds
#SBATCH --partition=GPUv100s
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH -o job_%j.out
#SBATCH -e job_%j.err
export CUDA_VISIBLE_DEVICES=0,1,2,3
module load cuda118
module load cuda118/toolkit/11.8.0
module load parallel
module load python/3.7.x-anaconda
source activate /archive/bioinformatics/DLLab/shared/CondaEnvironments/Aixa_ARMED_2

python run_AE_RE_allfolds.py
