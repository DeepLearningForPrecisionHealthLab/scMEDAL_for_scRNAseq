import os
import sys
# Define general data paths for all the simulation outputs

# Update this path to where you store your data
# /MyscMEDALExpt/Experiments/data/HealthyHeart_data


# To import data_base_path from paths_config
# Add the parent directory to the Python path
sys.path.append("../../")

# Now you can import from the parent directory
from paths_config import data_base_path,scenario_id

data_path = data_base_path

# Specify the folder containing the data to split
data2split_foldername = scenario_id


# Define the folder to save the splits, located inside data2split_foldername
# Ensure you create your own paths to prevent overwriting data
folder_splits = os.path.join(data2split_foldername, "splits")

