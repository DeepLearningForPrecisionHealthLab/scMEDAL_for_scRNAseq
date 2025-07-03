from typing import NamedTuple, List

class SAUCIEModelConfigs(NamedTuple):
    name:str="saucie"
    n_latent_dims:int=50
    layers:List[int]=[512, 132, 50]
    
    lambda_b:float=0.001
    lambda_c:float=0.
    lambda_d:float=0.
    learning_rate:float=0.0001
    num_iterations:int=50

    ignore:List[str]=['ignore', 'fold_list', "optimizer_configs", "loss_configs", "metric_configs",
    'batch_col', 'bio_col', 'donor_col', "layer_units_latent_classifier", "name",
    "monitor_metric", "stop_criteria", "get_pca", "get_baseline", 'use_z',
    'encoder_latent_name', 'sigmoid_eval_test', 'last_activation', 'get_pred',
    "eval_test", "optimizer", "loss", "loss_weights", "metrics"
    ]


