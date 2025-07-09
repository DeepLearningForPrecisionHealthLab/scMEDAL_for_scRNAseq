from typing import NamedTuple, List

class scMEDALREModelConfigs(NamedTuple):
    name:str="scmedalre"
    n_latent_dims:int=50
    layer_units:List[int]=[512,132]
    layer_units_classifier:List[int]=[5]
    n_pred:int=21
    n_clusters:int=19
    last_activation:str="linear"
    post_loc_init_scale:float=0.1
    prior_scale:float=0.25
    kl_weight:float=1e-5
    get_recon_cluster:bool=False
    ignore:List[str]=['ignore', 'fold_list', "optimizer_configs", "loss_configs", "metric_configs",
    "get_cf_batch", "compute_latents_callback", "n_components", "loss_recon", 
    "loss_multiclass", "metric_multiclass", "optimizer", "model_type", "tissue_col", 
    "batch_col", "bio_col", "donor_col", "layer_units_classifier", "get_recon_cluster", 
    "prior_scale", "post_loc_init_scale", "layer_units_latent_classifier", "n_pred", 
    "n_clusters", "name", "monitor_metric", "stop_criteria", "get_pca", "get_baseline", 
    "use_z", "encoder_latent_name", "sigmoid_eval_test", "last_activation", "get_pred", 
    "eval_test",
    ]