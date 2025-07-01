from typing import NamedTuple, Optional

class BaseScoreConfigs(NamedTuple):
    encoder_latent_name:str="latent"
    get_pca:bool=True
    n_components:int=2
    get_baseline:bool=False
    get_cf_batch:bool=False



class DataConfigs:
    valid_model_names=["ae", "aec","scmedalfe","scmedalfec", "scmedalre"]

    def __init__(self, model_name:Optional[str]=None):
        self.configs:NamedTuple = None
        
        if model_name is not None:
            self.load_configs(model_name)
    
    def load_configs(self, name:str):
        if name not in self.valid_model_names:
            raise ValueError(f"Invalid Name:{name}. See ModelConfigs.valid_names for acceptable inputs")

        if name == "ae":
            self.configs = self._load_ae_configs()
        elif name == "aec":
            self.configs = self._load_aec_configs()
        elif name == "scmedalfe":
            self.configs = self._load_scmedalfe_configs()
        elif name == "scmedalfec":
            self.configs = self._load_scmedalfec_configs()
        elif name == "scmedalre":
            self.configs = self._load_scmedalre_configs()

    def _load_ae_configs(self):
        configs = BaseScoreConfigs()
        return configs

    def _load_aec_configs(self):
        configs = BaseScoreConfigs()
        return configs

    def _load_scmedalfe_configs(self):
        configs = BaseScoreConfigs()
        return configs

    def _load_scmedalfec_configs(self):
        configs = BaseScoreConfigs()
        return configs
 
    def _load_scmedalre_configs(self):
        configs = BaseScoreConfigs()._replace(get_cf_batch=True)
        return configs
  