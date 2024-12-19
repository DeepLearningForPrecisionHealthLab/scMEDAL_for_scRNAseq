
import os

# Define general paths. Shared by experiment and model.

# 1. Define base paths (input data to the model)
data_base_path = "/archive/bioinformatics/DLLab/AixaAndrade/data/Genomic_data/VanGallen_2019"
scenario_id = "log_transformed_2916hvggenes"
input_base_path = os.path.join(data_base_path, scenario_id, 'splits')

# 2. General path to experiment results
outputs_path ="/archive/bioinformatics/DLLab/AixaAndrade/results/mixedeffectsdl/results/ARMED_genomics/VanGallen_2019/outputs"

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
#expt = "expt1_hpo_ae_da"
#expt = "expt1_hpo_aec_da"
#expt = "expt1_hpo_aec"
#expt = "expt1"
# expt = "expt1_hpo_aec_cce"
expt = "expt1_cce"
# Set to False if you did not run the model pipeline with get_pca=True for AEC
get_pca= True

if expt=="expt1":
    scaling="min_max"
    run_names_dict={"AE_RE":r"run_crossval_loss_recon_weight-110.0_loss_latent_cluster_weight-0.1_n_latent_dims-2_layer_units-512-132_kl_weight-0.0_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_2024-07-28_16-54",
                    "run_name_all":"leukemia_2916_genes_n_latent_dims-2"
                        }
    # Set this var to False if you are not planning to calculate clustering scores.
    calculate_clustering_scores = True
    # This is adding  the other models to the clustering scores table.
    # For other models than AE_RE: Using same expt than "expt_3_with_batch_norm_clusteringlosscurves_HP0_adjustment4_500epochs_linear_latentspace50"
    if calculate_clustering_scores:
            run_names_dict={"AE":r"run_crossval_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_compute_latents_callback-False_sample_size-10000_model_type-ae_n_components-2_2024-07-26_17-05",
                    # This is "AEC_reconloss10_classloss0.1"
                    "AEC":r"run_crossval_n_latent_dims-2_layer_units-512-132_n_pred-21_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_compute_latents_callback-False_sample_size-10000_model_type-aec_2024-07-28_17-07",
                    # This is AEC_DA_reconloss1500_classloss1
                    "AEC_DA":r"run_crossval_loss_gen_weight-1_loss_recon_weight-1500_loss_class_weight-1_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_model_type-ae_da_2024-07-26_17-58",
                    # This is "AE_DA_4000" (recon loss)
                    "AE_DA":r"run_crossval_loss_gen_weight-1_loss_recon_weight-4000_loss_class_weight-1_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_model_type-ae_da_2024-07-29_20-54",
                    "AE_RE":r"run_crossval_loss_recon_weight-110.0_loss_latent_cluster_weight-0.1_n_latent_dims-2_layer_units-512-132_kl_weight-0.0_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_2024-07-28_16-54",
                    "run_name_all":"leukemia_2916_genes_n_latent_dims-2"
                    }

    # Dictionary to hold the results paths
    model_names  = ["AE","AEC","AEC_DA","AE_DA","AE_RE"]
    results_path_dict = {}

    #for model_name,run_name in run_names_dict.items():
    for model_name,(run_name_key,run_name_value) in zip(model_names,run_names_dict.items()):
        if model_name == "run_name_all":
            pass   
        else:
            results_path_dict[run_name_key] = os.path.join(latent_space_path, model_name, run_name_value)

    # Dictionary to hold the results_path_dict_saved_models
    results_path_dict_saved_models = {}
    

    for model_name,(run_name_key,run_name_value) in zip(model_names,run_names_dict.items()):
        if model_name == "run_name_all":
            pass   
        else:
            if run_name_value:            
                results_path_dict_saved_models[run_name_key] = os.path.join(latent_space_path, model_name, run_name_value)




elif expt=="expt1_hpo_ae_da":
    # I realized that here I am using AEC with bce loss
    scaling="min_max"
    # Set this var to False if you are not planning to calculate clustering scores.
    calculate_clustering_scores = True
    # This is adding  the other models to the clustering scores table.
    # For other models than AE_RE: Using same expt than "expt_3_with_batch_norm_clusteringlosscurves_HP0_adjustment4_500epochs_linear_latentspace50"
    if calculate_clustering_scores:
        # varying the loss recon weight: loss_recon_weight
            run_names_dict={
                    "AE_DA_100":r"run_crossval_loss_gen_weight-1_loss_recon_weight-100_loss_class_weight-1_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_model_type-ae_da_2024-07-26_17-29",
                    "AE_DA_500":r"run_crossval_loss_gen_weight-1_loss_recon_weight-500_loss_class_weight-1_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_model_type-ae_da_2024-07-26_17-02",
                    "AE_DA_1000":r"run_crossval_loss_gen_weight-1_loss_recon_weight-1000_loss_class_weight-1_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_model_type-ae_da_2024-07-29_18-20",
                    "AE_DA_2000":r"run_crossval_loss_gen_weight-1_loss_recon_weight-2000_loss_class_weight-1_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_model_type-ae_da_2024-07-29_18-27",
                    "AE_DA_3000":r"run_crossval_loss_gen_weight-1_loss_recon_weight-3000_loss_class_weight-1_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_model_type-ae_da_2024-07-29_20-43",
                    "AE_DA_4000":r"run_crossval_loss_gen_weight-1_loss_recon_weight-4000_loss_class_weight-1_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_model_type-ae_da_2024-07-29_20-54",
                    "run_name_all":"leukemia_2916_genes_n_latent_dims-2_ae_da"
                    }
    # Dictionary to hold the results paths
    results_path_dict = {}

    for model_name,run_name in run_names_dict.items():
        if model_name == "run_name_all":
            pass   
        else:
            results_path_dict[model_name] = os.path.join(latent_space_path, "AE_DA", run_name)

elif expt=="expt1_hpo_aec_da":
    scaling="min_max"
    # Set this var to False if you are not planning to calculate clustering scores.
    calculate_clustering_scores = True
    # This is adding  the other models to the clustering scores table.
    # For other models than AE_RE: Using same expt than "expt_3_with_batch_norm_clusteringlosscurves_HP0_adjustment4_500epochs_linear_latentspace50"
    if calculate_clustering_scores:
        # varying the loss recon weight: loss_recon_weight
            run_names_dict={
                    "AEC_DA_reconloss500_classloss1":r"run_crossval_loss_gen_weight-1_loss_recon_weight-500_loss_class_weight-1_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_model_type-ae_da_2024-07-26_17-25",
                    "AEC_DA_reconloss1500_classloss1":r"run_crossval_loss_gen_weight-1_loss_recon_weight-1500_loss_class_weight-1_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_model_type-ae_da_2024-07-26_17-58",
                    "AEC_DA_reconloss1500_classloss2":r"run_crossval_loss_gen_weight-1_loss_recon_weight-1500_loss_class_weight-2_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_model_type-ae_da_2024-07-26_17-33",
                    "AEC_DA_reconloss2000_classloss2":r"run_crossval_loss_gen_weight-1_loss_recon_weight-2000_loss_class_weight-1_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_model_type-ae_da_2024-07-29_18-23",
                    "run_name_all":"leukemia_2916_genes_n_latent_dims-2_aec_da"
                    }
    # Dictionary to hold the results paths
    results_path_dict = {}

    for model_name,run_name in run_names_dict.items():
        if model_name == "run_name_all":
            pass   
        else:
            results_path_dict[model_name] = os.path.join(latent_space_path, "AEC_DA", run_name)


elif expt=="expt1_hpo_aec": #On this expt I used bce as class loss
    scaling="min_max"
    # Set this var to False if you are not planning to calculate clustering scores.
    calculate_clustering_scores = True
    # This is adding  the other models to the clustering scores table.
    # For other models than AE_RE: Using same expt than "expt_3_with_batch_norm_clusteringlosscurves_HP0_adjustment4_500epochs_linear_latentspace50"
    if calculate_clustering_scores:
        # varying the loss recon weight: loss_recon_weight
            run_names_dict={
                    "AEC_reconloss10_classloss0.1":"run_crossval_n_latent_dims-2_layer_units-512-132_n_pred-21_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_compute_latents_callback-False_sample_size-10000_model_type-aec_2024-07-28_17-07",
                    "AEC_reconloss1_classloss0.1":"run_crossval_n_latent_dims-2_layer_units-512-132_n_pred-21_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_compute_latents_callback-False_sample_size-10000_model_type-aec_2024-07-28_17-58",
                    "AEC_reconloss81_classloss0.1":"run_crossval_n_latent_dims-2_layer_units-512-132_n_pred-21_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_compute_latents_callback-False_sample_size-10000_model_type-aec_2024-07-28_16-18",
                    "run_name_all":"leukemia_2916_genes_n_latent_dims-2_aec"
                    }
    # Dictionary to hold the results paths
    results_path_dict = {}

    for model_name,run_name in run_names_dict.items():
        if model_name == "run_name_all":
            pass   
        else:
            results_path_dict[model_name] = os.path.join(latent_space_path, "AEC", run_name)


elif expt=="expt1_hpo_aec_cce":
    scaling="min_max"
    # Set this var to False if you are not planning to calculate clustering scores.
    calculate_clustering_scores = True
    # This is adding  the other models to the clustering scores table.
    # For other models than AE_RE: Using same expt than "expt_3_with_batch_norm_clusteringlosscurves_HP0_adjustment4_500epochs_linear_latentspace50"
    if calculate_clustering_scores:
        # varying the loss recon weight: loss_recon_weight
            run_names_dict={
                # selected AEC_reconloss100_classloss0.1
                    "AEC_reconloss100_classloss0.1":"run_crossval_n_latent_dims-2_layer_units-512-132_n_pred-21_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_compute_latents_callback-False_sample_size-10000_model_type-aec_2024-08-27_12-35",
                    "AEC_reconloss1_classloss0.1":"run_crossval_n_latent_dims-2_layer_units-512-132_n_pred-21_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_compute_latents_callback-False_sample_size-10000_model_type-aec_2024-08-27_11-41",
                    "AEC_reconloss10_classloss0.1":"run_crossval_n_latent_dims-2_layer_units-512-132_n_pred-21_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_compute_latents_callback-False_sample_size-10000_model_type-aec_2024-08-27_11-24",
                    "run_name_all":"leukemia_2916_genes_n_latent_dims-2_aec_cce"
                    }
    # Dictionary to hold the results paths
    results_path_dict = {}

    for model_name,run_name in run_names_dict.items():
        if model_name == "run_name_all":
            pass   
        else:
            results_path_dict[model_name] = os.path.join(latent_space_path, "AEC", run_name)





elif expt=="expt1_cce":
    # Note, the only thing that changed from expt1 was aec. On this version I am using aec with cce loss
    scaling="min_max"
    run_names_dict={"AE_RE":r"run_crossval_loss_recon_weight-110.0_loss_latent_cluster_weight-0.1_n_latent_dims-2_layer_units-512-132_kl_weight-0.0_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_2024-07-28_16-54",
                    "run_name_all":"leukemia_2916_genes_n_latent_dims-2_cce"
                        }
    # Set this var to False if you are not planning to calculate clustering scores.
    calculate_clustering_scores = True
    # This is adding  the other models to the clustering scores table.
    # For other models than AE_RE: Using same expt than "expt_3_with_batch_norm_clusteringlosscurves_HP0_adjustment4_500epochs_linear_latentspace50"
    if calculate_clustering_scores:
            run_names_dict={"AE":r"run_crossval_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_compute_latents_callback-False_sample_size-10000_model_type-ae_n_components-2_2024-07-26_17-05",
                    # selected AEC_reconloss100_classloss0.1
                    "AEC":r"run_crossval_n_latent_dims-2_layer_units-512-132_n_pred-21_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_compute_latents_callback-False_sample_size-10000_model_type-aec_2024-08-27_12-35",                    
                    # This is AEC_DA_reconloss1500_classloss1
                    "AEC_DA":r"run_crossval_loss_gen_weight-1_loss_recon_weight-1500_loss_class_weight-1_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_model_type-ae_da_2024-07-26_17-58",
                    # This is "AE_DA_4000" (recon loss)
                    "AE_DA":r"run_crossval_loss_gen_weight-1_loss_recon_weight-4000_loss_class_weight-1_n_latent_dims-2_layer_units-512-132_use_batch_norm-True_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_model_type-ae_da_2024-07-29_20-54",
                    "AE_RE":r"run_crossval_loss_recon_weight-110.0_loss_latent_cluster_weight-0.1_n_latent_dims-2_layer_units-512-132_kl_weight-0.0_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_2024-07-28_16-54",
                    "run_name_all":"leukemia_2916_genes_n_latent_dims-2_cce"
                    }

    # Dictionary to hold the results paths
    model_names  = ["AE","AEC","AEC_DA","AE_DA","AE_RE"]
    results_path_dict = {}

    #for model_name,run_name in run_names_dict.items():
    for model_name,(run_name_key,run_name_value) in zip(model_names,run_names_dict.items()):
        if model_name == "run_name_all":
            pass   
        else:
            results_path_dict[run_name_key] = os.path.join(latent_space_path, model_name, run_name_value)

    # Dictionary to hold the results_path_dict_saved_models
    results_path_dict_saved_models = {}
    

    for model_name,(run_name_key,run_name_value) in zip(model_names,run_names_dict.items()):
        if model_name == "run_name_all":
            pass   
        else:
            if run_name_value:            
                results_path_dict_saved_models[run_name_key] = os.path.join(latent_space_path, model_name, run_name_value)



# To copy this file
path2results_file= os.path.abspath(__file__)



