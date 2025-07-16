import os

# --------------------------------------------------------------------------------------
# Define general paths shared by the experiment and model
# --------------------------------------------------------------------------------------

# Get the directory of the current file
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define data base path relative to the current file's directory
data_base_path = os.path.join(base_dir, "../data/HealthyHeart_data")
print("data_base_path:", data_base_path)

scenario_id = "log_transformed_3000hvggenes"
input_base_path = os.path.join(data_base_path, scenario_id, 'splits')

# Define output paths
outputs_path = os.path.join(base_dir, "../outputs/HealthyHeart_outputs")
os.makedirs(outputs_path, exist_ok=True)
print("outputs_path:", outputs_path)

folder_name = scenario_id

latent_space_path = os.path.join(outputs_path, "latent_space", folder_name)
saved_models_path = os.path.join(outputs_path, "saved_models", folder_name)

# Path to compare_models (for outputs of compare models scripts)
compare_models_path = os.path.join(outputs_path, "compare_models", folder_name)

# --------------------------------------------------------------------------------------
# Experiment configuration


# Path to this file (for copying or reference)
path2results_file = os.path.abspath(__file__)
