import os

# --------------------------------------------------------------------------------------
# Define general paths shared by the experiment and model
# --------------------------------------------------------------------------------------

# Get the directory of the current file
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define data base path relative to the current file's directory
data_base_path = os.path.join(base_dir, "../data/AML_data")
print("data_base_path:", data_base_path)

scenario_id = "log_transformed_2916hvggenes"
input_base_path = os.path.join(data_base_path, scenario_id, 'splits')

# Define output paths
outputs_path = os.path.join(base_dir, "../outputs/AML_outputs")
# Ensure the directory is created when the module is imported
os.makedirs(outputs_path, exist_ok=True)
print("outputs_path:", outputs_path)

folder_name = scenario_id

latent_space_path = os.path.join(outputs_path, "latent_space", folder_name)
saved_models_path = os.path.join(outputs_path, "saved_models", folder_name)

# Path to compare_models (for outputs of compare models scripts)
compare_models_path = os.path.join(outputs_path, "compare_models", folder_name)

# --------------------------------------------------------------------------------------
# Experiment configuration
# --------------------------------------------------------------------------------------
expt = "expt_test"
get_pca = True  # Set to False if PCA was not computed for the models

if expt == "expt_test":
    scaling = "min_max"

    # Unique run names with timestamps should be provided here
    run_names_dict = {
        "scMEDAL-RE": "run_crossval_loss_recon_weight-110.0_loss_latent_cluster_weight-0.1_n_latent_dims-2_layer_units-512-132_kl_weight-0.0_scaling-min_max_batch_size-512_epochs-2_patience-30_sample_size-10000_2024-12-24_15-32",
        "run_name_all": "DefineGeneralname4yourexpt"
    }

    # Set True if you plan to calculate clustering scores
    calculate_clustering_scores = True

    # If calculating clustering scores, add other models
    if calculate_clustering_scores:
        run_names_dict.update({
            "AE": "run_crossval_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-2_patience-30_compute_latents_callback-False_sample_size-10000_model_type-ae_n_components-2_2024-12-24_14-30",
            "AEC": "run_crossval_n_latent_dims-2_layer_units-512-132_n_pred-21_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-2_patience-30_compute_latents_callback-False_sample_size-10000_model_type-aec_2024-12-24_14-45",
            "scMEDAL-FEC": "run_crossval_loss_gen_weight-1_loss_recon_weight-2000_loss_class_weight-1_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-2_patience-30_sample_size-10000_model_type-ae_da_2024-12-24_15-14",
            "scMEDAL-FE": "run_crossval_loss_gen_weight-1_loss_recon_weight-4000_loss_class_weight-1_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-2_patience-30_sample_size-10000_model_type-ae_da_2024-12-24_15-10"
        })

    # Dictionaries to hold paths to latent space results and saved model results
    results_path_dict = {}
    results_path_dict_saved_models = {}

    for model_name, run_name in run_names_dict.items():
        # Skip if it's the general run_name_all placeholder
        if model_name == "run_name_all":
            continue

        # Only add paths if run_name is not empty
        if run_name:
            results_path_dict[model_name] = os.path.join(latent_space_path, model_name, run_name)
            results_path_dict_saved_models[model_name] = os.path.join(saved_models_path, model_name, run_name)

# Path to this file (for copying or reference)
path2results_file = os.path.abspath(__file__)
