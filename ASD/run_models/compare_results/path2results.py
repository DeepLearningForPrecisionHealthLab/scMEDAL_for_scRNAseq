
import os

# Define general paths. Shared by experiment and model.

# 1. Define base paths (input data to the model)
data_base_path = "/archive/bioinformatics/DLLab/AixaAndrade/data/Genomic_data/ASD/reverse_norm"
scenario_id = "log_transformed_2916hvggenes"
input_base_path = os.path.join(data_base_path, scenario_id, 'splits')

# 2. General path to experiment results
outputs_path =outputs_path ="/archive/bioinformatics/DLLab/AixaAndrade/results/mixedeffectsdl/results/ARMED_genomics/ASD/reverse_norm/outputs"
folder_name = scenario_id
# Folder structure setup
# saved_models_base = 'path_to_saved_models'
latent_space_path = os.path.join(outputs_path,"latent_space", folder_name)
# latent_space_base = 'path_to_latent_space'
saved_models_path = os.path.join(outputs_path,"saved_models", folder_name)

# Convert paths to raw strings
latent_space_path = rf"{latent_space_path}"
saved_models_path = rf"{saved_models_path}"

# Path to compare_models: This is where I will save outputs of compare models scripts
compare_models_path = os.path.join(outputs_path, "compare_models", folder_name)

# Define experiment paths. This is because an experiment van be run multiple times with a different time stamp.
# Enter the names of the experiment you want to compare
# expt = "expt1"
expt="expt1_v1.9"
# expt = "expt1_hpo_ae_da"
# Set to False if you did not run the model pipeline with get_pca=True for AEC
get_pca= True

if expt=="expt1":
    scaling="min_max"
    run_names_dict={"AE_RE":r"run_crossval_loss_recon_weight-110.0_loss_latent_cluster_weight-0.1_n_latent_dims-2_layer_units-512-132_kl_weight-0.0_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_2024-07-28_23-18",
                    "run_name_all":"asd_2916_genes_n_latent_dims-2"
                        }
    # Set this var to False if you are not planning to calculate clustering scores.
    calculate_clustering_scores = True
    # This is adding  the other models to the clustering scores table.
    # For other models than AE_RE: Using same expt than "expt_3_with_batch_norm_clusteringlosscurves_HP0_adjustment4_500epochs_linear_latentspace50"
    if calculate_clustering_scores:
            run_names_dict={"AE":r"run_crossval_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_compute_latents_callback-False_sample_size-10000_model_type-ae_n_components-2_2024-07-28_22-56",
                    "AEC":r"run_crossval_n_latent_dims-2_layer_units-512-132_n_pred-17_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_compute_latents_callback-False_sample_size-10000_model_type-aec_2024-07-28_23-08",
                    "AEC_DA":r"run_crossval_loss_gen_weight-1_loss_recon_weight-1000_loss_class_weight-1_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_model_type-ae_da_2024-07-28_23-15",
                    # This is "AE_DA_1000", I changed the name to avoid problems in umap
                    "AE_DA":r"run_crossval_loss_gen_weight-1_loss_recon_weight-1000_loss_class_weight-1_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_model_type-ae_da_2024-07-29_10-35",
                    "AE_RE":r"run_crossval_loss_recon_weight-110.0_loss_latent_cluster_weight-0.1_n_latent_dims-2_layer_units-512-132_kl_weight-0.0_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_2024-07-28_23-18",
                    "run_name_all":"asd_2916_genes_n_latent_dims-2"
                    }

    # Dictionary to hold the results paths
    results_path_dict = {}

    for model_name,run_name in run_names_dict.items():
        if model_name == "run_name_all":
            pass
        elif "AE_DA" in model_name:
            results_path_dict[model_name] = os.path.join(latent_space_path, "AE_DA", run_name)
        else:
            results_path_dict[model_name] = os.path.join(latent_space_path, model_name, run_name)

    # Dictionary to hold the results_path_dict_saved_models
    results_path_dict_saved_models = {}

    for model_name,run_name in run_names_dict.items():
        if model_name == "run_name_all":
            pass
        elif "AE_DA" in model_name:
            results_path_dict[model_name] = os.path.join(latent_space_path, "AE_DA", run_name)
        else:
            if run_name:            
                results_path_dict_saved_models[model_name] = os.path.join(saved_models_path, model_name, run_name)


elif expt=="expt1_hpo_ae_da":
    scaling="min_max"
    # Set this var to False if you are not planning to calculate clustering scores.
    calculate_clustering_scores = True
    # This is adding  the other models to the clustering scores table.
    # For other models than AE_RE: Using same expt than "expt_3_with_batch_norm_clusteringlosscurves_HP0_adjustment4_500epochs_linear_latentspace50"
    if calculate_clustering_scores:
        # varying the loss recon weight: loss_recon_weight
            run_names_dict={
                    "AE_DA_10":r"run_crossval_loss_gen_weight-1_loss_recon_weight-10_loss_class_weight-1_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_model_type-ae_da_2024-07-28_23-42",
                    "AE_DA_100":r"run_crossval_loss_gen_weight-1_loss_recon_weight-100_loss_class_weight-1_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_model_type-ae_da_2024-07-28_23-03",
                    "AE_DA_1000":r"run_crossval_loss_gen_weight-1_loss_recon_weight-1000_loss_class_weight-1_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_model_type-ae_da_2024-07-29_10-35",
                    "run_name_all":"asd_2916_genes_n_latent_dims-2_ae_da"
                    }
    # Dictionary to hold the results paths
    results_path_dict = {}

    for model_name,run_name in run_names_dict.items():
        if model_name == "run_name_all":
            pass   
        else:
            results_path_dict[model_name] = os.path.join(latent_space_path, "AE_DA", run_name)

elif expt=="expt1_v1.9":
    # Note, the only thing that changed from expt1 was aec. On this version I am using aec with cce loss
    scaling="min_max"
    run_names_dict={"AE_RE":r"run_crossval_loss_recon_weight-110.0_loss_latent_cluster_weight-0.1_n_latent_dims-2_layer_units-512-132_kl_weight-0.0_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_2024-07-28_23-18",
                    "run_name_all":"asd_2916_genes_n_latent_dims-2_v1.9"
                        }
    # Set this var to False if you are not planning to calculate clustering scores.
    calculate_clustering_scores = True
    # This is adding  the other models to the clustering scores table.
    # For other models than AE_RE: Using same expt than "expt_3_with_batch_norm_clusteringlosscurves_HP0_adjustment4_500epochs_linear_latentspace50"
    if calculate_clustering_scores:
            run_names_dict={"AE":r"run_crossval_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_compute_latents_callback-False_sample_size-10000_model_type-ae_n_components-2_2024-07-28_22-56",
                    "AEC":r"run_crossval_n_latent_dims-2_layer_units-512-132_n_pred-17_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_compute_latents_callback-False_sample_size-10000_model_type-aec_2024-08-27_12-43",
                    "AEC_DA":r"run_crossval_loss_gen_weight-1_loss_recon_weight-1000_loss_class_weight-1_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_model_type-ae_da_2024-07-28_23-15",
                    # This is "AE_DA_1000", I changed the name to avoid problems in umap
                    "AE_DA":r"run_crossval_loss_gen_weight-1_loss_recon_weight-1000_loss_class_weight-1_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_model_type-ae_da_2024-07-29_10-35",
                    "AE_RE":r"run_crossval_loss_recon_weight-110.0_loss_latent_cluster_weight-0.1_n_latent_dims-2_layer_units-512-132_kl_weight-0.0_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_2024-07-28_23-18",
                    "run_name_all":"asd_2916_genes_n_latent_dims-2_v1.9"
                    }

    # Dictionary to hold the results paths
    results_path_dict = {}

    for model_name,run_name in run_names_dict.items():
        if model_name == "run_name_all":
            pass
        elif "AE_DA" in model_name:
            results_path_dict[model_name] = os.path.join(latent_space_path, "AE_DA", run_name)
        else:
            results_path_dict[model_name] = os.path.join(latent_space_path, model_name, run_name)

    # Dictionary to hold the results_path_dict_saved_models
    results_path_dict_saved_models = {}

    for model_name,run_name in run_names_dict.items():
        if model_name == "run_name_all":
            pass
        elif "AE_DA" in model_name:
            results_path_dict[model_name] = os.path.join(latent_space_path, "AE_DA", run_name)
        else:
            if run_name:            
                results_path_dict_saved_models[model_name] = os.path.join(saved_models_path, model_name, run_name)

# To copy this file
path2results_file= os.path.abspath(__file__)

