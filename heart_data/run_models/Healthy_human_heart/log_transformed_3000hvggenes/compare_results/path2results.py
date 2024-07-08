
import os

# Define general paths. Shared by experiment and model.

# Genral path to experiment results
outputs_path ="/archive/bioinformatics/DLLab/AixaAndrade/results/mixedeffectsdl/results/ARMED_genomics/heart_data/outputs"
# dataset_type = "subsets"
folder_name = "Healthy_human_heart_data/log_transformed_3000hvggenes"
# Folder structure setup
# saved_models_base = 'path_to_saved_models'
latent_space_path = os.path.join(outputs_path, "saved_models", folder_name)
# latent_space_base = 'path_to_latent_space'
saved_models_path = os.path.join(outputs_path, "latent_space", folder_name)

# Convert paths to raw strings
latent_space_path = rf"{latent_space_path}"
saved_models_path = rf"{saved_models_path}"

# Path to compare_models: This is where I will save outputs of compare models scripts
compare_models_path = os.path.join(outputs_path, "compare_models", folder_name)

# Define experiment paths. This is because an experiment van be run multiple times with a different time stamp.
# Enter the names of the experiment you want to compare
expt = "expt3_batch_cf"

# Set to False if you did not run the model pipeline with get_pca=True for AEC
get_pca= True

if expt=="expt3_batch_cf":
    run_names_dict={"AE_RE":r"run_crossval_loss_recon_weight-110.0_loss_latent_cluster_weight-0.1_n_latent_dims-2_layer_units-512-132_kl_weight-0.0_batch_size-512_epochs-500_patience-30_compute_latents_callback-False_sample_size-10000_get_cf_batch-True_2024-06-12_11-56",
                    "run_name_all":"healthyheart_3000_genes_n_latent_dims-2_layer_units-512-132_epochs-1_batch_cf"
                        }
    # Set this var to False if you are not planning to calculate clustering scores.
    calculate_clustering_scores = True
    # This is adding  the other models to the clustering scores table.
    # For other models than AE_RE: Using same expt than "expt_3_with_batch_norm_clusteringlosscurves_HP0_adjustment4_500epochs_linear_latentspace50"
    if calculate_clustering_scores:
            run_names_dict={"AE":r"run_crossval_n_latent_dims:2_layer_units:[512 132]_use_batch_norm:True_batch_size:512_epochs:500_patience:30_compute_latents_callback:True_sample_size:10000_model_type:ae_n_components:50_2024-05-16_01-54",
                    "AEC":r"run_crossval_n_latent_dims:2_layer_units:[512 132]_n_pred:13_use_batch_norm:True_batch_size:512_epochs:500_patience:30_compute_latents_callback:True_sample_size:10000_model_type:aec_2024-05-15_23-46",
                    "AEC_DA":r"run_crossval_loss_gen_weight:1_loss_recon_weight:9450_loss_class_weight:1_n_latent_dims:2_layer_units:[512 132]_use_batch_norm:True_batch_size:512_epochs:500_patience:30_compute_latents_callback:True_sample_size:10000_model_type:ae_da_2024-05-16_00-06",
                    "AE_DA":r"run_crossval_loss_gen_weight:1_loss_recon_weight:5400_loss_class_weight:1_n_latent_dims:2_layer_units:[512 132]_use_batch_norm:True_batch_size:512_epochs:500_patience:30_compute_latents_callback:True_sample_size:10000_model_type:ae_da_2024-05-15_23-42",
                    "AE_RE":r"run_crossval_loss_recon_weight:110.0_loss_latent_cluster_weight:0.1_n_latent_dims:2_layer_units:[512 132]_kl_weight:0.0_batch_size:512_epochs:500_patience:30_compute_latents_callback:True_sample_size:10000_2024-05-15_23-44",
                    "AE_RE":r"run_crossval_loss_recon_weight-110.0_loss_latent_cluster_weight-0.1_n_latent_dims-2_layer_units-512-132_kl_weight-0.0_batch_size-512_epochs-500_patience-30_compute_latents_callback-False_sample_size-10000_get_cf_batch-True_2024-06-12_11-56",
                    "run_name_all":"healthyheart_3000_genes_n_latent_dims-2_layer_units-512-132_epochs-1_batch_cf"
                    }

    # Dictionary to hold the results paths
    results_path_dict = {}

    for model_name,run_name in run_names_dict.items():
        if model_name == "run_name_all":
            pass   
        else:
            results_path_dict[model_name] = os.path.join(latent_space_path, model_name, run_name)

    # Dictionary to hold the results_path_dict_saved_models
    results_path_dict_saved_models = {}

    for model_name,run_name in run_names_dict.items():
        if model_name == "run_name_all":
            pass   
        else:
            if run_name:            
                results_path_dict_saved_models[model_name] = os.path.join(saved_models_path, model_name, run_name)




