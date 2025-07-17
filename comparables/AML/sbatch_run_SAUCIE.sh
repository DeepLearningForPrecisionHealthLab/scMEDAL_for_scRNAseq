#!/bin/bash

#SBATCH --job-name=SAUCIE
#SBATCH --partition=GPUp4
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err
export CUDA_VISIBLE_DEVICES=0
module load cuda90/9.0.176
module load cuda91/toolkit/9.1.85
module load cudnn/7.0 
module load parallel
module load python/3.7.x-anaconda

# Update the path to the environment you use to run your script
source activate /archive/bioinformatics/DLLab/shared/CondaEnvironments/Aixa_SAUCIE

python run_SAUCIE.py
