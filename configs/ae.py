from typing import NamedTuple, List

class AEModelConfigs(NamedTuple):
    name:str="ae"
    n_latent_dims:int=2
    layer_units:List[int]=[512,132]
    last_activation:str="linear"
    use_batch_norm:bool=True
    ignore:List[str]=['ignore', 'fold_list', "optimizer_configs", "loss_configs", "metric_configs",
    'batch_col', 'bio_col', 'donor_col', "layer_units_latent_classifier", "name",
    "monitor_metric", "stop_criteria", "get_pca", "get_baseline", 'use_z',
    'encoder_latent_name', 'sigmoid_eval_test', 'last_activation', 'get_pred',
    "eval_test", "optimizer", "loss", "loss_weights", "metrics","configs-BaseScoreConfigs","get_cf_batch",
    ]


