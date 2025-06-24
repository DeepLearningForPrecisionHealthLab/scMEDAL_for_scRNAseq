from typing import Optional, Dict, Any
from utils.model_train_utils import generate_run_name
import configs.configs as cfg
import tensorflow as tf


class Model:
    valid_models=["ae"]

    def __init__(self, model:str, **kwargs):
        assert model in self.valid_models, f"Invalid model {model}"
        self.model = model
        self.compile_configs:cfg.CompileConfigs=self._init_compile_configs(kwargs)
        self.model_configs:cfg.ModelConfigs=self._init_model_configs(kwargs)
        self.data_configs:cfg.DataConfigs=self._init_data_configs(kwargs)
        self.training_configs:cfg.TrainingConfig=self._init_training_configs(kwargs)
        self.scores_configs:cfg.ScoreConfig=self._init_score_configs(kwargs)
        self.expt_design_configs:cfg.ExperimentDesignConfigs=self._init_exp_design_configs(kwargs)
        self.model_params:Dict[str,Any] = self.__init_modelparams()

    def _init_compile_configs(self, kwargs:Optional[Dict[str, Any]]=None):
        self.compile_configs = cfg.CompileConfigs()        
        if kwargs is not None:
            self.compile_configs = self.compile_configs._replace(**{k:v for k,v in kwargs.items() if k in self.compile_configs._fields})
        self.__replace_tf_aliases()
        return self.compile_configs

    def _init_model_configs(self, kwargs:Optional[Dict[str, Any]]=None):
        self.model_configs = cfg.ModelConfigs(self.model)        
        if kwargs is not None:
            self.model_configs = self.model_configs.configs._replace(**{k:v for k,v in kwargs.items() if k in self.model_configs.configs._fields})
        return self.model_configs
      
    def _init_data_configs(self, kwargs:Optional[Dict[str, Any]]=None):
        self.data_configs = cfg.DataConfigs()        
        if kwargs is not None:
            self.data_configs = self.data_configs._replace(**{k:v for k,v in kwargs.items() if k in self.data_configs._fields})
        return self.data_configs
    
    def _init_training_configs(self, kwargs:Optional[Dict[str, Any]]=None):
        self.training_configs = cfg.TrainingConfigs(self.model)        
        if kwargs is not None:
            self.training_configs = self.training_configs._replace(**{k:v for k,v in kwargs.items() if k in self.training_configs._fields})
        return self.training_configs

    def _init_score_configs(self, kwargs:Optional[Dict[str, Any]]=None):
        self.score_configs = cfg.ScoreConfigs()        
        if kwargs is not None:
            self.score_configs = self.score_configs._replace(**{k:v for k,v in kwargs.items() if k in self.score_configs._fields})
        return self.score_configs

    def _init_exp_design_configs(self, kwargs:Optional[Dict[str, Any]]=None):
        self.expt_design_configs = cfg.ExperimentDesignConfigs()        
        if kwargs is not None:
            self.expt_design_configs = self.expt_design_configs._replace(**{k:v for k,v in kwargs.items() if k in self.expt_design_configs._fields})
        return self.expt_design_configs

    def __replace_tf_aliases(self):
        self.compile_configs._replace(**{
            "optimizer":tf.keras.optimizers.get(self.compile_configs.optimizer),
            "loss":[tf.keras.losses.get(loss) for loss in self.compile_configs.loss],
            "metrics":[tf.keras.metrics.get(metric) for metric in self.compile_configs.metrics]
        })

    def __init_modelparams(self):
        self.model_params = {
            **self.compile_configs._asdict(),
            **self.model_configs._asdict(),
            **self.data_configs._asdict(),
            **self.training_configs._asdict(),
            **self.scores_configs._asdict(),
            **self.expt_design_configs._asdict(), 
        }
        ignore = self.model_params.get("ignore", [])
        self.model_params.pop("ignore")
        self.model_params['run_name'] = generate_run_name(self.model_params, ignore, name="run_crossval") 
        return self.model_params
        




