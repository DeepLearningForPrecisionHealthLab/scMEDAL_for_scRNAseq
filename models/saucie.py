from typing import Optional, Dict, Any
import configs.configs as cfg

from .base import Model
from .SAUCIE.model import SAUCIE as SAUCIE_alg
from .SAUCIE.model import Loader

class SAUCIE(Model):
    def __init__(self, **kwargs):
        super().__init__(model_name="saucie", **kwargs)
        self.alg = SAUCIE_alg

    #def run_train(self, data_path:Optional[str]=None, outputs_path:Optional[str]=None, named_experiment:Optional[str]=None, save_model:bool=True, quick:bool=False, plotconfigs:Optional[cfg.PlotConfigs]=None, plot_kwargs:Optional[Dict[str, Any]]=None): 


