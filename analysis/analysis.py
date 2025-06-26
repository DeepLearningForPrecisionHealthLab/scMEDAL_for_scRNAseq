import os
import json
from typing import Optional, List, Dict, Union, NamedTuple
from abc import ABC, abstractmethod
from utils.defaults import OUTPUTS_DIR, DEFAULTS_ABS_PATH, AML_DATA_DIR, AML_EXPERIMENT_NAME

from .AML.compare_clustering_scores import compare_clustering_scores_AML
from .AML.genomap_and_plot import genomap_and_plot


def dict_to_named_tuple(d:Dict, name:Optional[str]=None) -> NamedTuple:
    return NamedTuple(f"{name}", [(k,v) for k,v in d.items()])
    
class AnalysisPaths(NamedTuple):
    experiment_name:str
    analysis_name:str
    data_path:str
    data_folder_name:str
    splits_path:str
    outputs_path:str
    model_results_folder_dict:Dict[str,str]
    latent_space_path:Dict[str,str]
    saved_models_path:Dict[str,str]
    
class Analysis(ABC):
    def __init__(self, experiment_name:Optional[str], model_result_folder_dict:Optional[Dict[str,str]]=None, analysis_name:Optional[str]=None) -> None:
        self.experiment_name = experiment_name
        self.model_result_folder_dict = model_result_folder_dict
        self.analysis_name = analysis_name

    @abstractmethod
    def clustering_scores(self):
        raise NotImplementedError()
    
    @abstractmethod
    def genomap(self):
        raise NotImplementedError()
    
    @abstractmethod
    def umap(self):
        raise NotImplementedError()

    def analysis_named_experiment_paths(self, experiment_name:str, model_result_folder_dict:Dict[str,str], analysis_name:Optional[str]=None) -> AnalysisPaths:
        if model_result_folder_dict is None:
            return None
            
        valid_named_experiment=["AML","ASD", "HH"]
        assert experiment_name in valid_named_experiment, f"Unrecognized experiment name {experiment_name}. Valid experiments include {valid_named_experiment}"
        paths = {
            "experiment_name":experiment_name,
            "analysis_name":analysis_name if analysis_name is not None else ""
        }
        if experiment_name == "AML":
            paths["data_path"] = f"/archive/bioinformatics/DLLab/AixaAndrade/src/gitfront/scMEDAL_for_scRNAseq/Experiments/data/AML_data"
            paths['data_folder_name'] = "log_transformed_2916hvggenes"
        elif experiment_name == "ASD":
            paths["data_path"] = f"/archive/bioinformatics/DLLab/AixaAndrade/src/gitfront/scMEDAL_for_scRNAseq/Experiments/data/ASD_data/reverse_norm/"
            paths['data_folder_name'] = "log_transformed_2916hvggenes"
        elif experiment_name == "HH":
            paths["data_path"] = f"/archive/bioinformatics/DLLab/AixaAndrade/src/gitfront/scMEDAL_for_scRNAseq/Experiments/data/HealthyHeart_data/"
            paths['data_folder_name'] = "log_transformed_3000hvggenes"
        paths['splits_path'] = os.path.join( paths["data_path"],  paths['data_folder_name'], "splits")
        paths["outputs_path"] = os.path.join(OUTPUTS_DIR, experiment_name, "compare_models", paths["data_folder_name"])
        paths["model_results_folder_dict"] = model_result_folder_dict
        latent_space_header = os.path.join(OUTPUTS_DIR, experiment_name, "latent_space", paths["data_folder_name"])
        model_saved_header = os.path.join(OUTPUTS_DIR, experiment_name, "saved_models", paths["data_folder_name"])
        paths["latent_space_path"] = {k:os.path.join(latent_space_header, k, v) for k,v in model_result_folder_dict.items() }
        paths["saved_models_path"] = {k:os.path.join(model_saved_header, k, v) for k,v in model_result_folder_dict.items() }
        
        return AnalysisPaths(**paths)

class AMLAnalysis(Analysis):
    def __init__(self, model_result_folder_dict:Optional[Dict[str,str]]=None, analysis_name:Optional[str]=None, experiment_name:str="AML", *args, **kwargs) -> None:
        super().__init__(experiment_name, model_result_folder_dict, analysis_name)
        self.paths = self.analysis_named_experiment_paths(experiment_name, model_result_folder_dict, analysis_name)

    def _check_assertions(self, model_result_folder_dict:Optional[Dict[str,str]]=None):
        assert model_result_folder_dict is not None or self.model_result_folder_dict is not None, "Model result folders must be provided to run analysis."
        if model_result_folder_dict is not None:
            self.paths = self.analysis_named_experiment_paths(self.experiment_name, model_result_folder_dict, self.analysis_name)
        assert self.paths.saved_models_path is not None, "A dictionary of saved model paths must be provided."
        assert self.paths.latent_space_path is not None, "A dictionary of latent space paths must be provided."
        assert set(self.paths.latent_space_path.keys()) == set(self.paths.saved_models_path.keys()), "Latent space paths must have same keys as model paths."
        #assert set(self.paths.saved_models_path.keys()) == set(self.paths.model_results_folder_dict.keys()), "Saved Model paths must have same keys as model paths."

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
            model_result_folder_dict:Optional[Dict[str,str]]=None,
            dataset_type:str="test",
        ):
        self._check_assertions(model_result_folder_dict)
                
        compare_models_path = self.paths.outputs_path
        run_names_dict = self.paths.model_results_folder_dict
        results_path_dict = self.paths.latent_space_path
        results_path_saved_models = self.paths.saved_models_path
        model_configs_dict = {}
        for model, model_folder_id in run_names_dict.items():
            model_configs_dict[model] = self._get_pca_from_configs(results_path_saved_models[model])
        run_names_dict["run_name_all"] = self.paths.analysis_name

        compare_clustering_scores_AML(
            run_names_dict=run_names_dict, 
            results_path_dict=results_path_dict, 
            compare_models_path=compare_models_path, 
            dataset_type=dataset_type,
            models2process_dict=model_configs_dict
        )

    
    def _aml_genomap_vars(self):
        # --------------------------------------------------------------------------------------
        # 3. Define variables
        # --------------------------------------------------------------------------------------
        celltype = ["Mono", "Mono-like"]
        n_cells_per_batch = 300
        n_batches = 19

        # Define number of genes in the genomap = colNum * rowNum
        n_genes = 2916
        colNum = 54
        rowNum = 54
        batches_to_select_from = ["AML420B", "BM5", "MUTZ3"]

        ### This is for AML
        gene_index_col = "Gene"

    def genomap(self,
            model_folder_dict:Dict[str,str],
            experimentid:Optional[str]=None,
        ):
        self._check_assertions()
        
        

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
