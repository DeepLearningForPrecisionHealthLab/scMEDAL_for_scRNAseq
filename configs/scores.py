from typing import NamedTuple, Optional

class ScoreConfigs(NamedTuple):
    encoder_latent_name:Optional[str]=None
    get_pca:bool=True
    n_components:int=2
    get_baseline:bool=False