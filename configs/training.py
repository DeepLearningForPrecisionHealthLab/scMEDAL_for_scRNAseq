from typing import NamedTuple, Union, List, Dict, Any, Optional
# from tensorflow.keras.optimizers import Optimizer
# from tensorflow.keras.losses import Loss
# from tensorflow.keras.metrics import Metric
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.losses import MeanSquaredError as mse_loss
from tensorflow.keras.losses import CategoricalCrossentropy as cce_loss
from tensorflow.keras.metrics import MeanSquaredError as mse_metric
from tensorflow.keras.metrics import CategoricalAccuracy as cce_metric

class TrainingConfigs(NamedTuple):
    model_type:Optional[str]=None
    batch_size:int=512
    epochs:int=500
    monitor_metric:str="val_loss"
    patience:int=30
    stop_criteria:str="early_stopping"
    compute_latents_callback:bool=False
    model_type:Optional[str]=None
    ##### Should this even be here?!?!
    sample_size:int=10_000


class CompileConfigs:
    valid_model_names=["ae", "aec"]

    def __init__(self, model_name:Optional[str]=None):
        self.configs:Dict[str, Any] = None
        
        if model_name is not None:
            self.load_configs(model_name)
    
    def load_configs(self, name:str):
        if name not in self.valid_model_names:
            raise ValueError(f"Invalid Name:{name}. See ModelConfigs.valid_names for acceptable inputs")

        if name == "ae":
            self.configs = self._load_ae_configs()
        elif name == "aec":
            self.configs = self._load_aec_configs()


    def _load_ae_configs(self) -> Dict[str, Any]:
        return { 
                "optimizer": Adam(lr=0.0001),  # Optimizer
                "loss": [mse_loss(name='mean_squared_error')],  # Loss function
                "loss_weights": 1,  # Weight for loss components
                "metrics": [[mse_metric()]]  # Metrics for evaluation
            }

    def _load_aec_configs(self) -> Dict[str, Any]:
        return {
            "optimizer": Adam(lr=0.0001),  # Optimizer
            "loss": {
                'reconstruction_output': mse_loss(name='mse'),  # Reconstruction loss
                'classification_output': cce_loss(name='cce')   # Classification loss
            },
            "loss_weights": {
                'reconstruction_output': 100,   # Weight for reconstruction loss
                'classification_output': 0.1   # Weight for classification loss
            },
            "metrics": {
                'reconstruction_output': [mse_metric(name="mse_metric")],
                'classification_output': [cce_metric(name='cce_metric')]
            }
        }
