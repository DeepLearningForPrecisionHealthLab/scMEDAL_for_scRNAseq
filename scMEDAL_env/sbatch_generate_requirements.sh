#!/bin/bash

#SBATCH --job-name=get_env_req
#SBATCH --partition=128GB
#SBATCH --nodes=1
#SBATCH --ntasks=1


module load python/3.7.x-anaconda
# To generate requirements file for Aixa_scDML env
# source activate /archive/bioinformatics/DLLab/shared/CondaEnvironments/Aixa_scDML
# python generate_requirements_Aixa_scDML.py

# To generate requirements file for Aixa_genomap env
source activate /project/bioinformatics/DLLab/shared/CondaEnvironments/Aixa_genomap
python generate_requirements_Aixa_genomap.py

# To generate requirements file for ARMED_Aixa_v2 env
# source activate /project/bioinformatics/DLLab/shared/CondaEnvironments/Aixa_ARMED_2
# python generate_requirements_ARMED_Aixa_v2.py
