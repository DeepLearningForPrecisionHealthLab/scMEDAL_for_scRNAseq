from .ae import AE
from .aec import AEC
from .scmedalfe import scMEDALFE
from .scmedalfec import scMEDALFEC
from .scmedalre import scMEDALRE
from .saucie import SAUCIE
from .mec import MEC
from .base import Model

from utils.defaults import AML_MODEL_KWARGS, ASD_MODEL_KWARGS, HH_MODEL_KWARGS

from typing import Optional, Dict, Any, Union

model_aliases = {
    "AE":AE, "ae":AE,
    "AEC":AEC, "aec":AEC,
    "scMEDAL-FE":scMEDALFE, "scmedalfe":scMEDALFE, 
    "scMEDAL-FEC":scMEDALFEC, "scmedalfec":scMEDALFEC, 
    "scMEDAL-RE":scMEDALRE, "scmedalre":scMEDALRE,
    "SAUCIE":SAUCIE, "saucie":SAUCIE,     
    "MEC":MEC, "mec":MEC,
}

named_experiments = {
    "AML":"Acute Myeloid Leukemia",
    "ASD":"Autism Spectrum Disorder",
    "HH":"Healthy Heart"
}

def _parse_model_kwargs_for_named_experiment(model_name:str, model_kwargs:Dict[str,Any], experiment_name:str):
    def __check_update_specific_model_kwargs_for_named_experiment(model_name:str,experiment_name:str):
        updated_kwargs = {}
        #####################################
        ############## AML ##################
        #####################################
        if experiment_name == "AML" and (model_name  == "scmedalfe" or model_name == "scMEDAL-FE"):
            updated_kwargs['loss_recon_weight'] = 250
            updated_kwargs["loss_gen_weight"] = 1
        if experiment_name == "AML" and (model_name  == "scmedalfec" or model_name == "scMEDAL-FEC"):
            updated_kwargs['loss_recon_weight'] = 2_000
            updated_kwargs["loss_gen_weight"] = 1
            updated_kwargs["loss_class_weight"] = 1
        if experiment_name == "AML" and (model_name  == "aec" or model_name == "AEC"):
            updated_kwargs['loss_weights'] = {'reconstruction_output': 100,'classification_output': 0.1}
        if experiment_name == "AML" and (model_name  == "scmedalre" or model_name == "scMEDAL-RE"):
            updated_kwargs['loss_recon_weight'] = 110
            updated_kwargs["loss_latent_cluster_weight"] = 0.1

        #####################################
        ############## ASD ##################
        #####################################
        if experiment_name == "ASD" and (model_name  == "aec" or model_name == "AEC"):
            updated_kwargs['loss_weights'] = {'reconstruction_output': 100,'classification_output': 0.1}
        if experiment_name == "ASD" and (model_name  == "scmedalfe" or model_name == "scMEDAL-FE"):
            updated_kwargs['loss_recon_weight'] = 1_000
            updated_kwargs["loss_gen_weight"] = 1
        if experiment_name == "ASD" and (model_name  == "scmedalfec" or model_name == "scMEDAL-FEC"):
            updated_kwargs['loss_recon_weight'] = 2_000
            updated_kwargs["loss_gen_weight"] = 1
            updated_kwargs["loss_class_weight"] = 1
        if experiment_name == "ASD" and (model_name  == "scmedalre" or model_name == "scMEDAL-RE"):
            updated_kwargs['loss_recon_weight'] = 110
            updated_kwargs["loss_latent_cluster_weight"] = 0.1
        
        #####################################
        ############## HH ###################
        #####################################
        if experiment_name == "HH" and (model_name  == "aec" or model_name == "AEC"):
            updated_kwargs['loss_weights'] = {'reconstruction_output': 100,'classification_output': 0.1}
        if experiment_name == "HH" and (model_name  == "scmedalfe" or model_name == "scMEDAL-FE"):
            updated_kwargs["loss_gen_weight"] = 1
            updated_kwargs['loss_recon_weight'] = 600
        if experiment_name == "HH" and (model_name  == "scmedalfec" or model_name == "scMEDAL-FEC"):
            updated_kwargs['loss_recon_weight'] = 2_000
            updated_kwargs["loss_gen_weight"] = 1
            updated_kwargs["loss_class_weight"] = 1
        if experiment_name == "HH" and (model_name  == "scmedalre" or model_name == "scMEDAL-RE"):
            updated_kwargs['loss_recon_weight'] = 110
            updated_kwargs["loss_latent_cluster_weight"] = 0.1
        
    
        return updated_kwargs

    # Jank way of updating the default parameters.
    _update_kwarg_defaults = __check_update_specific_model_kwargs_for_named_experiment(model_name=model_name, experiment_name=experiment_name)
    for kwarg, val in _update_kwarg_defaults.items():
        model_kwargs[kwarg] = model_kwargs.get(kwarg, val) 

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
    model_kwargs = _parse_model_kwargs_for_named_experiment(model_name=model_name, model_kwargs=model_kwargs, experiment_name=named_experiment)

    mod = model_aliases[model_name](**model_kwargs)
    mod.run_train(named_experiment=named_experiment) if train_kwargs is None else mod.run_train(named_experiment=named_experiment, **train_kwargs)
    return mod
