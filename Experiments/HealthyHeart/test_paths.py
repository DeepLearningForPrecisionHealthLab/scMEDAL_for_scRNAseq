import os

# Get the directory of the current file
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define your paths relative to the current file's directory

data_base_path = os.path.join(base_dir,"../data/HealthyHeart_data")

print("data_base_path:",data_base_path)

outputs_path = os.path.join(base_dir,"../outputs/HealthyHeart_outputs")

print("outputs_path",outputs_path)
