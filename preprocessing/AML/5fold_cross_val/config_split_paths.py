import os
import sys
from pathlib import Path
ROOT_PATH  = Path.cwd().resolve().parents[2]  # two levels up from AML dir (scMEDAL_for_scRNAseq)
sys.path.insert(0, str(ROOT_PATH ))
print(ROOT_PATH )
# Define general data paths for all the simulation outputs

from utils.defaults import AML_DATA_DIR,AML_EXPERIMENT_NAME 


data_path = AML_DATA_DIR#os.path.join(AML_DATA_DIR, "adata_merged")

# Specify the folder containing the data to split
data2split_foldername = AML_EXPERIMENT_NAME 


# Define the folder to save the splits, located inside data2split_foldername
# Ensure you create your own paths to prevent overwriting data
folder_splits = os.path.join(data2split_foldername, "splits")



