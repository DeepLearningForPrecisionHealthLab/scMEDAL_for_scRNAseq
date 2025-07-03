import os
import json
from typing import Optional, List, Dict, Union, NamedTuple
from abc import ABC, abstractmethod
from utils.defaults import OUTPUTS_DIR, DEFAULTS_ABS_PATH, AML_DATA_DIR, AML_EXPERIMENT_NAME

from .AML.compare_clustering_scores import compare_clustering_scores_AML
from .AML.genomap_and_plot import genomap_and_plot
from .AML.compare_results_umap import get_umap


from dataclasses import dataclass, field
import random

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


@dataclass
class GenomapConfig:
    # core paths
    compare_models_path: str
    data_base_path: str
    scenario_id: str
    input_base_path: str
    analysis_name: Optional[str] = None

    # tuning knobs
    celltype: Optional[List[str]] = None
    batches:  Optional[List[str]] = None
    # genomap
    n_cells_per_batch: int = 300
    n_batches: int = None
    n_genes: int = 2916
    n_col: int = 54
    n_row: int = 54
    num_iter: int = 2 #100 

    cell_id_col : str = None
    gene_index_col: str = None
    scaling: str = "min_max"
    add_inputs_fe: bool = True
    extra_recon: str = "fe"   # "fe", "all", or "none"
    seed: int = 42
    n_cells_2_plot: int = 4
    n_top_genes : int = 10
    min_val : int = -1
    max_val : int = 2
    # derived
    n_inputs_fe: int = field(init=False)

    def __post_init__(self):
        self.n_inputs_fe = {"fe": 2, "all": 5}.get(self.extra_recon, 0)
        random.seed(self.seed)


class AMLAnalysis(Analysis):
    def __init__(self, model_result_folder_dict:Optional[Dict[str,str]]=None, analysis_name:Optional[str]=None, experiment_name:str="AML", *args, **kwargs) -> None:
        super().__init__(experiment_name, model_result_folder_dict, analysis_name)
        self.paths = self.analysis_named_experiment_paths(experiment_name, model_result_folder_dict, analysis_name)

    def _check_update_assertions(self, model_result_folder_dict:Optional[Dict[str,str]]=None):
        assert model_result_folder_dict is not None or self.model_result_folder_dict is not None, "Model result folders must be provided to run analysis."
        if model_result_folder_dict is not None:
            self.paths = self.analysis_named_experiment_paths(self.experiment_name, model_result_folder_dict, self.analysis_name)
        assert self.paths.saved_models_path is not None, "A dictionary of saved model paths must be provided."
        assert self.paths.latent_space_path is not None, "A dictionary of latent space paths must be provided."
        assert set(self.paths.latent_space_path.keys()) == set(self.paths.saved_models_path.keys()), "Latent space paths must have same keys as model paths."

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

    def _clustering_scores_kwargs(self) -> Dict[str,str]:
        return {
            "run_names_dict":self.paths.model_results_folder_dict,
            "results_path_dict":self.paths.latent_space_path,
            "compare_models_path":self.paths.outputs_path,
            "analysis_name":self.paths.analysis_name
        }

    def clustering_scores(self, 
            model_result_folder_dict:Optional[Dict[str,str]]=None,
            dataset_type:str="test",
        ):
        self._check_update_assertions(model_result_folder_dict)
                
        model_configs_dict = {}
        for model, model_folder_id in model_result_folder_dict.items():
            model_configs_dict[model] = self._get_pca_from_configs(self.paths.saved_models_path[model])

        compare_clustering_scores_AML(
            **self._clustering_scores_kwargs(),
            dataset_type=dataset_type,
            models2process_dict=model_configs_dict
        )

    def _genomap_kwargs(self):
        return {
            "run_names_dict":self.paths.model_results_folder_dict,
            "results_path_dict":self.paths.latent_space_path,
            "compare_models_path":self.paths.outputs_path,
            "data_base_path":self.paths.data_path,
            "scenario_id":self.paths.data_folder_name,
            "input_base_path":self.paths.splits_path,
            "analysis_name":self.paths.analysis_name
        }

    def genomap(self,
            model_result_folder_dict:Optional[Dict[str,str]]=None,
            celltype:Optional[List[str]]=None,
            batches:Optional[List[str]]=None,
            n_cells_per_batch:int=300,
            n_batches:int=19,
            n_genes:int=2916,
            n_col:int=54,
            n_row:int=54,
            gene_index_col:str="Gene",
            scaling:str="min_max",
            models:Optional[List[str]]=None, # Default is to do all models in model_result_folder_dict.
            types:Optional[List[str]]=None,  # ["train", "test", "val"]
            splits:Optional[List[int]]=None, # i.e. [1, 2] ...
            add_inputs_fe:bool=True, 
            extra_recon:str="fe", # "fe" or "all" ( Not really sure what this does.)
            seed:int=42, # random seed
            n_cells_2_plot: int = 4,
            n_top_genes : int = 10,
            min_val : int = None,
            max_val : int = None,
            num_iter: int = 100 ,
            cell_id_col : str = None,

        ):
        # This will update self.paths which is the basis for _genomap_kwargs
        # Not ideal, but workable.
        self._check_update_assertions(model_result_folder_dict)
        
                # 1. Build GenomapConfig from core paths + tuning knobs
        core = self._genomap_kwargs()
        cfg = GenomapConfig(
            compare_models_path = core["compare_models_path"],
            data_base_path      = core["data_base_path"],
            scenario_id         = core["scenario_id"],
            input_base_path     = core["input_base_path"],
            analysis_name       = core["analysis_name"],
            celltype            = celltype,
            batches             = batches,
            n_cells_per_batch   = n_cells_per_batch,
            n_batches           = n_batches,
            n_genes             = n_genes,
            n_col               = n_col,
            n_row               = n_row,
            num_iter            = num_iter,
            cell_id_col         = cell_id_col,
            gene_index_col      = gene_index_col,
            scaling             = scaling,
            add_inputs_fe       = add_inputs_fe,
            extra_recon         = extra_recon,
            seed                = seed,
            n_cells_2_plot      = n_cells_2_plot,
            n_top_genes         = n_top_genes,
            min_val             = min_val,
            max_val             = max_val,
        )

        # 2. Delegate to legacy wrapper that expects (run_names_dict, results_path_dict, cfg, ?)
        return genomap_and_plot(
            core["run_names_dict"],
            core["results_path_dict"],
            cfg,
            models=models,
            types=types,
            splits=splits,
        )

        if celltype is None:
            celltype = ["Mono", "Mono-like"]
        
        genomap=None
        try:
            genomap = genomap_and_plot(
                **self._genomap_kwargs(),
                celltype=celltype,
                batches=batches,
                n_cells_per_batch=n_cells_per_batch,
                n_batches=n_batches,
                n_genes=n_genes,
                n_col=n_col,
                n_row=n_row,
                gene_index_col=gene_index_col,
                scaling=scaling,
                models=models,
                types=types,
                splits=splits,
                add_inputs_fe=add_inputs_fe,
                extra_recon=extra_recon,
                seed=rng_seed
                )
        except Exception as e:
            raise e

        return genomap

    
    def _umap_kwargs(self):
        return {
            "run_names_dict":self.paths.model_results_folder_dict,
            "results_path_dict":self.paths.latent_space_path,
            "compare_models_path":self.paths.outputs_path,
            "input_base_path":self.paths.splits_path,
            "analysis_name":self.paths.analysis_name
        }

    def umap(self, model_result_folder_dict:Optional[Dict[str,str]]=None, **kwargs):
        # Not ideal, but workable.
        self._check_update_assertions(model_result_folder_dict)
    
        umap = None
        try:
            umap = get_umap(
                **self._umap_kwargs(),
                **kwargs
            )
        except Exception as e:
            raise e
        return umap