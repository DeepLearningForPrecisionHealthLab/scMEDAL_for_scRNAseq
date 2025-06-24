from typing import Optional,NamedTuple
from .ae import AEModelConfigs


class ModelConfigs:
    valid_names=["ae"]

    def __init__(self, model_name:Optional[str]=None):
        self.configs:NamedTuple = None
        
        if model_name is not None:
            self.load_configs(model_name)
    
    def load_configs(self, name:str):
        if name not in self.valid_names:
            raise ValueError(f"Invalid Name:{name}. See ModelConfigs.valid_names for acceptable inputs")

        if name == "ae":
            self.configs = AEModelConfigs()
