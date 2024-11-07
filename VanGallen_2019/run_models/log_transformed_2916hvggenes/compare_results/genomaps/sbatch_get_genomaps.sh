#!/bin/bash

#SBATCH --job-name=get_genomaps
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
source activate /project/bioinformatics/DLLab/shared/CondaEnvironments/Aixa_genomap


# python get_single_genomap_multibatch.py
python pipeline_CMmultibatch_genomap_and_plot_multicell.py
