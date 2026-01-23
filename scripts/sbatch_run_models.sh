#!/bin/bash

#SBATCH --job-name=scMEDAL
#SBATCH --partition=GPUv100s
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH -o job_%j.out
#SBATCH -e job_%j.err
#export CUDA_VISIBLE_DEVICES=0
module load cuda118
module load cuda118/toolkit/11.8.0
module load parallel
module load python/3.7.x-anaconda

# Note: This slurm script may not be necessary for you, you can run the scripts directly from your terminal
# Update the path to the environment you use to run your script
source activate /archive/bioinformatics/DLLab/shared/CondaEnvironments/scMEDAL

# To run scMEDAL subnetworks and ablation tests
# python 1-run_scMEDAL_alldatasets.py

# To run AEC batch classifier only
# python 1-run_scMEDAL_alldatasets_AECbatch.py

# To run mec (Random Forest) classifier experiement
# python 2-mec_aml_patientgroup.py
# python 2-mec_asd_dx.py
# python 2-mec_hh_tissue.py


# python 2-mec_aml_celltype.py
# python 2-mec_asd_celltype.py
# python 2-mec_hh_celltype.py





