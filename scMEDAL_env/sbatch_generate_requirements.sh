#!/bin/bash

#SBATCH --job-name=get_env_req
#SBATCH --partition=128GB
#SBATCH --nodes=1
#SBATCH --ntasks=1


module load python/3.7.x-anaconda
# To generate requirements file for Aixa_scvi env
# source activate /archive/bioinformatics/DLLab/shared/CondaEnvironments/Aixa_scvi
# python generate_requirements_scVI.py


# To generate requirements file for scMEDAL
# source activate /archive/bioinformatics/DLLab/shared/CondaEnvironments/scMEDAL
# python generate_requirements_scMEDAL.py
