from typing import NamedTuple, Union, List, Dict, Any, Optional
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric

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

class CompileConfigs(NamedTuple):
    """ Will use built in `get` method if string provided """
    optimizer:Union[str,Optimizer]="Adam"
    #optimizer_configs:Optional[Dict[str,Any]]={}#None#{"learning_rate":0.001}
    loss:List[Union[str,Loss]]=["MeanSquaredError"]
    #loss_configs:Optional[Dict[str,Any]]={}#None#{"loss_weights":1}
    metrics:List[Union[str,Metric]]=["MeanSquaredError"]
    #metric_configs:Optional[Dict[str,Any]]={}#None




