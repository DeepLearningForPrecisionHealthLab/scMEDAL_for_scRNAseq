
import os

models = ["AE", "AEC", "scMEDALFE", "scMEDALFEC", "scMEDALRE"]
tmp = [f"""#!/bin/bash

#SBATCH --job-name={mod}
#SBATCH --partition=GPUv100s
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err
export CUDA_VISIBLE_DEVICES=0
module load cuda118
module load cuda118/toolkit/11.8.0
module load parallel

# Update the path to the environment you use to run your script
cd /archive/bioinformatics/DLLab/src/AustinMarckx/git/scMEDAL_for_scRNAseq
source activate /archive/bioinformatics/DLLab/shared/CondaEnvironments/scMEDAL

python <<HEREDOC
import models.models as mods
mods.{mod}().run_train(named_experiment="AML")
HEREDOC
""" for mod in models]


if __name__ == "__main__":
    print("UNCOMMENT ME")

    # tmp_scripts = []
    # for idx, script in enumerate(tmp):
    #     curr_script_path = f"tmp/tmp{idx}.sh"
    #     tmp_scripts.append(curr_script_path)
    #     with open(curr_script_path, "w") as f:
    #         f.writelines(script)
        
    # for script in tmp_scripts:
    #     os.system(f"sbatch {script}")