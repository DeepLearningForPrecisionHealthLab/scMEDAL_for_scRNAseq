from typing import NamedTuple, List

class AECModelConfigs(NamedTuple):
    name:str="aec"
<<<<<<< HEAD
    n_latent_dims:int=2
=======
    n_latent_dims:int=50
>>>>>>> bc7d766fb90c6d45c716908e51471d864b7ebff1
    layer_units:List[int]=[512,132]
    layer_units_latent_classifier:List[int]=[2]
    n_pred:int=21
    last_activation:str="linear"
    use_batch_norm:bool=True
    ignore:List[str]=['ignore', 'fold_list', "optimizer_configs", "loss_configs", "metric_configs",
    "n_components", 'batch_col', 'bio_col', 'donor_col', "layer_units_latent_classifier",
    "name", "monitor_metric", "stop_criteria", "get_pca", "get_baseline", 'use_z',
    'encoder_latent_name', 'sigmoid_eval_test', 'last_activation', 'get_pred',
    "eval_test", "optimizer", "loss", "loss_weights", "metrics"
    ]
