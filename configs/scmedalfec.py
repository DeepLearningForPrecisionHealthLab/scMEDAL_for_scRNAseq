from typing import NamedTuple, List

class scMEDALFECModelConfigs(NamedTuple):
    name:str="scmedalfec"
    n_latent_dims:int=2
    layer_units:List[int]=[512,132]
    layer_units_latent_classifier:List[int]=[2]
    n_pred:int=21
    n_clusters:int=19
    last_activation:str="linear"
    use_batch_norm:bool=True
    ignore:List[str]=['ignore', 'fold_list', "optimizer_configs", "loss_configs", "metric_configs",
    "compute_latents_callback", "n_components", "batch_col", "bio_col", "donor_col",
    "loss_recon", "loss_multiclass", "metric_multiclass", "opt_autoencoder", "opt_adversary",
    "layer_units_latent_classifier", "n_pred", "n_clusters", "name", "monitor_metric",
    "stop_criteria", "get_pca", "get_baseline", "use_z", "encoder_latent_name",
    "sigmoid_eval_test", "last_activation", "get_pred", "eval_test"
    ]