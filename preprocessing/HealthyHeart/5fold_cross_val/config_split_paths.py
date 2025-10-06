# Update this path to where you store your data
# /scMEDAL_for_scRNAseq/Experiments/data/HealthyHeart_data


import os
import sys
from pathlib import Path

ROOT_PATH  = Path.cwd().resolve().parents[2]  # two levels up from HH dir (scMEDAL_for_scRNAseq)
sys.path.insert(0, str(ROOT_PATH ))
print(ROOT_PATH )
# Define general data paths for all the simulation outputs

# Update this path to where you store your data

from utils.defaults import HH_DATA_DIR,HH_EXPERIMENT_NAME 

data_path = HH_DATA_DIR

# Specify the folder containing the data to split
data2split_foldername = HH_EXPERIMENT_NAME 


# Define the folder to save the splits, located inside data2split_foldername
# Ensure you create your own paths to prevent overwriting data
folder_splits = os.path.join(data2split_foldername, "splits")


