from .ae import AE
from .aec import AEC
from .scmedalfe import scMEDALFE
from .scmedalfec import scMEDALFEC
from .scmedalre import scMEDALRE
from .base import Model

from utils.defaults import AML_MODEL_KWARGS, ASD_MODEL_KWARGS, HH_MODEL_KWARGS

from typing import Optional, Dict, Any, Union

model_aliases = {
    "AE":AE, "ae":AE,
    "AEC":AEC, "aec":AEC,
    "scMEDAL-FE":scMEDALFE, "scmedalfe":scMEDALFE, 
    "scMEDAL-FEC":scMEDALFEC, "scmedalfec":scMEDALFEC, 
    "scMEDAL-RE":scMEDALRE, "scmedalre":scMEDALRE,
}

named_experiments = {
    "AML":"Acute Myeloid Leukemia",
    "ASD":"Autism Spectrum Disorder",
    "HH":"Healthy Heart"
}

def _parse_model_kwargs_for_named_experiment(model_kwargs:Dict[str,Any], experiment_name:str):
    if experiment_name == "AML":
        model_kwargs = {**model_kwargs, **AML_MODEL_KWARGS}
    elif experiment_name == "ASD":
        model_kwargs = {**model_kwargs, **ASD_MODEL_KWARGS}
    elif experiment_name == "HH":
        model_kwargs = {**model_kwargs, **HH_MODEL_KWARGS}
    return model_kwargs

def train_model_on_named_experiment(model_name:str, named_experiment:str, model_kwargs:Optional[Dict[str,Any]]=None, train_kwargs:Optional[Dict[str,Any]]=None) -> Model:
    assert model_name in model_aliases.keys(), f"Invalid model name. Please select one from: {model_aliases.keys()}"
    assert named_experiment in named_experiments, f"Invalid Experiment name. Please select one from: {named_experiments.keys()}"
    
    if model_kwargs is None:
        model_kwargs = {}
    model_kwargs = _parse_model_kwargs_for_named_experiment(model_kwargs=model_kwargs, experiment_name=named_experiment)

    mod = model_aliases[model_name](**model_kwargs)
    mod.run_train(named_experiment=named_experiment) if train_kwargs is None else mod.run_train(named_experiment=named_experiment, **train_kwargs)
    return mod
