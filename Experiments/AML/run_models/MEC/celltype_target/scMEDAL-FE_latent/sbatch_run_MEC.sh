#!/bin/bash

#SBATCH --job-name=MEC
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
#source activate /archive/bioinformatics/DLLab/shared/CondaEnvironments/Aixa_ARMED_2
source activate /project/bioinformatics/DLLab/shared/CondaEnvironments/Aixa_ARMED_2

# Make sure the output path exists
#output_path="/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics/heart_data/outputs/log/Healthy_human_heart_data/log_transformed/AEC/"

# Now use the variable in the srun command
# This script requires exactly two command-line arguments. One for intFold/split number (int) and another one for GPU card (int)\n. 
# Example to run from terminal: python test_script.py 0 0 
#srun --exclusive -N1 -n1 -c 16 --output="${output_path}${SLURM_JOB_ID}_AEC_ALLFOLDS.txt" python run_AEC_allfolds.py &
python mec_allfolds.py

