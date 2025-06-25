import os
import json
from typing import Optional, List, Dict, Union
from abc import ABC, abstractmethod
from utils.defaults import OUTPUTS_DIR, DEFAULTS_ABS_PATH, AML_DATA_DIR, AML_EXPERIMENT_NAME

from .AML.compare_clustering_scores import compare_clustering_scores_AML
from .AML.genomap_and_plot import genomap_and_plot

class Analysis(ABC):
    valid_named_experiment=["AML","ASD", "HH"]
    def __init__(self, saved_model_paths:Optional[Dict[str,str]]=None, latent_space_paths:Optional[Dict[str,str]]=None, *args, **kwargs) -> None:
        self.model_paths = saved_model_paths
        self.latent_space_paths = latent_space_paths

    def load_named_experiment_paths(self, experiment:str) -> Dict[str, str]:
        assert experiment in self.valid_named_experiment, f"Unrecognized experiment name {experiment}. Valid experiments include {self.valid_named_experiment}"
        paths = None
        if experiment == "AML":
            paths = {}
            paths["data_path"] = f"/archive/bioinformatics/DLLab/AixaAndrade/src/gitfront/scMEDAL_for_scRNAseq/Experiments/data/AML_data"
            paths['scenario_id'] = "log_transformed_2916hvggenes"
            paths['splits_folder'] = "splits"
        elif experiment == "ASD":
            paths = {}
            paths["data_path"] = f"/archive/bioinformatics/DLLab/AixaAndrade/src/gitfront/scMEDAL_for_scRNAseq/Experiments/data/ASD_data/reverse_norm/"
            paths['scenario_id'] = "log_transformed_2916hvggenes"
            paths['splits_folder'] = "splits"
        elif experiment == "HH":
            paths = {}
            paths["data_path"] = f"/archive/bioinformatics/DLLab/AixaAndrade/src/gitfront/scMEDAL_for_scRNAseq/Experiments/data/HealthyHeart_data/"
            paths['scenario_id'] = "log_transformed_3000hvggenes"
            paths['splits_folder'] = "splits"

        return paths

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
    aml_relative_path = os.path.join(os.getcwd(),"analysis", "AML")
    _aml_clustering_score_subdir = os.path.join(os.getcwd(),"analysis", "AML", "clustering_scores")
    _aml_genomap_subdir = os.path.join(os.getcwd(),"analysis", "AML", "genomaps")
    _aml_umap_subdir = os.path.join(os.getcwd(),"analysis", "AML", "umap_plots")
    aml_outputs_default_path = os.path.join(OUTPUTS_DIR, "AML", "compare_models")

    def __init__(self, outputs_path:Optional[str]=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.compare_models_path = outputs_path if outputs_path is not None else self.aml_outputs_default_path

    def _check_assertions(self):
        assert self.model_paths is not None, "A dictionary of saved model paths must be provided."
        assert self.latent_space_paths is not None, "A dictionary of latent space paths must be provided."
        assert set(self.latent_space_paths.keys()) == set(self.model_paths.keys()), "Latent space paths must have same keys as model paths."

    def load_configs(self, path:str, config_filename:str="configs.json"):
        fp = os.path.join(path, config_filename)
        with open(fp, "r") as f:
            configs = json.load(f)
        return configs

    def _get_pca_from_configs(self, configspath:str) -> str:
        configs = self.load_configs(configspath)
        pca_val = configs.get("get_pca")
        if pca_val == True:
            return "preprocess_results_model_pca_format"
        return "process_single_model_format"


    def clustering_scores(self, 
            model_folder_dict:Dict[str,str],
            experimentid:Optional[str]=None,
            dataset_type:str="test",
        ):
        self._check_assertions()
        run_names_dict = {k:v.split(os.path.sep)[-2] for k, v in self.model_paths.items()}
        results_path_dict = self.latent_space_paths
        results_path_saved_models = self.model_paths
        models2process_dict = {}
        for model, model_folder_id in run_names_dict.items():
            results_path_dict[model] = os.path.join(self.latent_space_paths[model], model_folder_dict[model])
            results_path_saved_models[model] = os.path.join(self.model_paths[model], model_folder_dict[model])
            models2process_dict[model] = self._get_pca_from_configs(results_path_saved_models[model])
        run_names_dict['run_name_all'] = experimentid if experimentid is not None else ""

        compare_clustering_scores_AML(
            run_names_dict=run_names_dict, 
            results_path_dict=results_path_dict, 
            compare_models_path=self.compare_models_path, 
            dataset_type=dataset_type,
            models2process_dict=models2process_dict
        )

    
    def genomap(self,
            model_folder_dict:Dict[str,str],
            experimentid:Optional[str]=None,
        ):
        self._check_assertions()
        
        run_names_dict = {k:v.split(os.path.sep)[-2] for k, v in self.model_paths.items()}
        results_path_dict = self.latent_space_paths
        results_path_saved_models = self.model_paths
        for model, model_folder_id in run_names_dict.items():
            results_path_dict[model] = os.path.join(self.latent_space_paths[model], model_folder_dict[model])
            results_path_saved_models[model] = os.path.join(self.model_paths[model], model_folder_dict[model])
        
        run_names_dict['run_name_all'] = experimentid if experimentid is not None else ""

        paths = self.load_named_experiment_paths("AML")
        data_path = paths.get("data_path")
        folder_name = paths.get("scenario_id")
        splits_path = os.path.join(data_path, folder_name, paths.get("splits_folder"))

        return genomap_and_plot(
            run_names_dict=run_names_dict,
            results_path_dict=results_path_dict,
            compare_models_path=self.compare_models_path,
            data_base_path=data_path,
            scenario_id=folder_name,
            input_base_path=splits_path,
            path2results_file=DEFAULTS_ABS_PATH,
            scaling="min_max", ## ugly yes, but I've never seen this not used and I'm out of time.
        )

    
    def umap(self):
        self._check_assertions()
    
        pass
