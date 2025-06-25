import os

from typing import Optional, List, Dict, Union
from abc import ABC, abstractmethod

from utils.defaults import OUTPUTS_DIR

class Analysis(ABC):
    def __init__(self, saved_model_paths:Optional[Dict[str,str]]=None, latent_space_paths:Optional[Dict[str,str]]=None, *args, **kwargs) -> None:
        self.model_paths = saved_model_paths
        self.latent_space_paths = latent_space_paths

    @abstractmethod
    def clustering_scores(self):
        raise NotImplementedError()
    
    @abstractmethod
    def genomap(self):
        raise NotImplementedError()
    
    @abstractmethod
    def umap(self):
        raise NotImplementedError()


class AMLAnalysis(Analysis):
    aml_outputs_default_path = os.path.join(OUTPUTS_DIR, "AML", "compare_models")

    def __init__(self, outputs_path:Optional[str]=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _check_assertions(self):
        assert self.model_paths is not None, "A dictionary of saved model paths must be provided."
        assert self.latent_space_paths is not None, "A dictionary of latent space paths must be provided."
        assert set(self.latent_space_paths.keys()) == set(self.model_paths.keys()), "Latent space paths must have same keys as model paths."


    def clustering_scores(self, 
            experimentid:Optional[str]=None,
            dataset_type:str="test",

        ):
        self._check_assertions()
        run_names_dict = {k:v.split(os.path.sep)[-2] for k, v in self.model_paths.items()}
        run_names_dict['run_name_all'] = experimentid if experimentid is not None else ""
        results_path_dict = {}
        results_path_saved_models = {}
        for model, model_folder_id in run_names_dict.items():
            results_path_dict[model] = os.path.join(self.latent_space_paths[model], model, model_folder_id)
            results_path_saved_models[model] = os.path.join(self.model_paths[model], model, model_folder_id)

    
    def genomap(self):
        self._check_assertions()

        pass
    
    def umap(self):
        self._check_assertions()
    
        pass
