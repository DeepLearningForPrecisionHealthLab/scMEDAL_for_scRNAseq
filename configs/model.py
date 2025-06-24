from typing import Optional,NamedTuple
from .ae import AEModelConfigs
from .aec import AECModelConfigs
from .scmedalfe import scMEDALFEModelConfigs
from .scmedalfec import scMEDALFECModelConfigs


class ModelConfigs:
    valid_model_names=["ae", "aec", "scmedalfe", "scmedalfec"]

    def __init__(self, model_name:Optional[str]=None):
        self.configs:NamedTuple = None
        
        if model_name is not None:
            self.load_configs(model_name)
    
    def load_configs(self, name:str):
        if name not in self.valid_model_names:
            raise ValueError(f"Invalid Name:{name}. See ModelConfigs.valid_names for acceptable inputs")

        if name == "ae":
            self.configs = AEModelConfigs()
        elif name == "aec":
            self.configs = AECModelConfigs()
        elif name == "scmedalfe":
            self.configs = scMEDALFEModelConfigs()
        elif name == "scmedalfec":
            self.configs = scMEDALFECModelConfigs()
