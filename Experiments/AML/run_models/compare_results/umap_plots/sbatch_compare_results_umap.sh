#!/bin/bash

#SBATCH --job-name=get_umaps
#SBATCH --partition=384GB
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
# Update the path to the environment you use to run your script
# source activate /path/to/preprocess_and_plot_umaps_env
source activate /archive/bioinformatics/DLLab/shared/CondaEnvironments/Aixa_scDML

#python compare_results_umap.py
python compare_results_umap_50dims.py
