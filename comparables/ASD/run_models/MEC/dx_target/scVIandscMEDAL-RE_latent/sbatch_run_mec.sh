#!/bin/bash

#SBATCH --job-name=scVI
#SBATCH --partition=GPUp4
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err
export CUDA_VISIBLE_DEVICES=0


# module load cuda/12.1  # or the module version matching your PyTorch installation
# cuda121/toolkit/12.1.0 
# module load cudnn/8.6.0.163
# module load parallel
module load python/3.7.x-anaconda

# Update the path to the environment you use to run your script
source activate /archive/bioinformatics/DLLab/shared/CondaEnvironments/Aixa_scvi
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python mec_allfolds.py