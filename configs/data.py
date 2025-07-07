from typing import NamedTuple, List, Optional


class DataConfigs:
    valid_model_names=["ae", "aec","scmedalfe","scmedalfec", "scmedalre", "mec"]

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
        elif name == "mec":
            self.configs = self._load_mec_configs()

    def _load_ae_configs(self):
        configs = BaseDataConfigs()
        return configs

    def _load_aec_configs(self):
        configs = BaseDataConfigs()._replace(get_pred=True)
        return configs

    def _load_scmedalfe_configs(self):
        configs = BaseDataConfigs()._replace(use_z=True)
        return configs

    def _load_scmedalfec_configs(self):
        configs = BaseDataConfigs()._replace(use_z=True)
        return configs
 
    def _load_scmedalre_configs(self):
        configs = BaseDataConfigs()._replace(use_z=True)
        return configs
    
    def _load_mec_configs(self):
        configs = BaseDataConfigs()
        return configs



class BaseDataConfigs(NamedTuple):
    eval_test:bool=True
    use_z:bool=False
    get_pred:bool=False
    scaling:str="min_max"
    fold_list:List[int]=list(range(1,6))
    