from typing import NamedTuple, List, Dict

class MECModelConfigs(NamedTuple):
    name:str="mec"
<<<<<<< HEAD
    n_latent_dims:int=50
    layer_units:List[int]=[8,4]
=======
    n_latent_dims:int=5 #This is not an Autoencoder it's a classifier so dim 5 is fine
    layer_units:List[int]=[25,10]
>>>>>>> bc7d766fb90c6d45c716908e51471d864b7ebff1
    n_pred:int=21
    ignore:List[str]=['ignore', 'fold_list', "optimizer_configs", "loss_configs", "metric_configs",
    "n_components", 'batch_col', 'bio_col', 'donor_col', "layer_units_latent_classifier",
    "name", "monitor_metric", "stop_criteria", "get_pca", "get_baseline", 'use_z',
    'encoder_latent_name', 'sigmoid_eval_test', 'last_activation', 'get_pred',
    "eval_test", "optimizer", "loss", "loss_weights", "metrics", "latent_path_dict", "model_params", 
    "base_path", "add_re_2_mec_class", "batch_col_categories", "bio_col_categories",
<<<<<<< HEAD
    "models_list", "return_metrics", "return_adata_dict", "return_trained_model", "model_type", "compute_latents_callback"
=======
    "models_list", "return_metrics", "return_adata_dict", "return_trained_model", "model_type", "compute_latents_callback","get_cf_batch","patience","batch_size","layer_units","n_latent_dims"
>>>>>>> bc7d766fb90c6d45c716908e51471d864b7ebff1
    ]

    # Old LatentClassifier_config
    latent_keys_config:Dict[str,str]= {'fe_latent':'X_pca'}
    return_metrics:bool= True
    return_adata_dict:bool= True
    return_trained_model:bool= True
    model_type:str= 'mec'
    seed:int=42  # Fixed seed for reproducibility
    ## old Load Latent spaces dict
    latent_path_dict:str = None
    model_params:Dict[str,str]= None
    base_path:str= None
    models_list:List[str]= ["scmedalre", "scmedalfe", "scmedalfec", "aec", "ae"]
    batch_col_categories:str= None
    bio_col_categories:str=None