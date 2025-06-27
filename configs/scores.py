from typing import NamedTuple

class ScoreConfigs(NamedTuple):
    encoder_latent_name:str="latent"
    get_pca:bool=True
    n_components:int=2
    get_baseline:bool=False