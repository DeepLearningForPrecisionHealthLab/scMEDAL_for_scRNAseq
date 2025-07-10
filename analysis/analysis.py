import os
import json
from typing import Optional, List, Dict, Union, NamedTuple
from abc import ABC, abstractmethod
from utils.defaults import OUTPUTS_DIR, DEFAULTS_ABS_PATH, AML_DATA_DIR, AML_EXPERIMENT_NAME

from .compare_clustering_scores import compare_clustering_scores
from .genomap_and_plot import genomap_and_plot
from .compare_results_umap import get_umap


from dataclasses import dataclass, field
import random
import inspect 


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
    issparse: bool = False
    n_cells_2_plot: int = 4
    n_top_genes : int = 10
    min_val : int = -1
    max_val : int = 2
    # derived
    n_inputs_fe: int = field(init=False)

    def __post_init__(self):
        self.n_inputs_fe = {"fe": 2, "all": 5}.get(self.extra_recon, 0)
        random.seed(self.seed)


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
    def _clustering_scores_kwargs(self):
        raise NotImplementedError()

    @abstractmethod
    def _genomap_kwargs(self):
        raise NotImplementedError()

    @abstractmethod
    def _umap_kwargs(self):
        raise NotImplementedError()

    def _check_update_assertions(self, model_result_folder_dict:Optional[Dict[str,str]]=None):
        assert model_result_folder_dict is not None or self.model_result_folder_dict is not None, "Model result folders must be provided to run analysis."
        if model_result_folder_dict is not None:
            self.paths = self.analysis_named_experiment_paths(self.experiment_name, model_result_folder_dict, self.analysis_name)
        assert self.paths.saved_models_path is not None, "A dictionary of saved model paths must be provided."
        assert self.paths.latent_space_path is not None, "A dictionary of latent space paths must be provided."
        assert set(self.paths.latent_space_path.keys()) == set(self.paths.saved_models_path.keys()), "Latent space paths must have same keys as model paths."

    def clustering_scores(self, 
            model_result_folder_dict:Optional[Dict[str,str]]=None,
            dataset_type:str="test",
        ):
        import numpy as np
        self._check_update_assertions(model_result_folder_dict)
                
        model_configs_dict = {}
        for model, _folder_id in model_result_folder_dict.items():
            try:
                model_configs_dict[model] = self._get_pca_from_configs(
                    self.paths.saved_models_path[model]
                )
            except Exception as exc:
                print(f"[WARN] {model}: {exc}  using single-model format (AE will default to PCA format if present)")
                model_configs_dict[model] = "process_single_model_format"

                # and, while we are in the fallback block, make sure AE
                # still gets the PCA version.
                if "AE" in model_result_folder_dict:
                    model_configs_dict["AE"] = "preprocess_results_model_pca_format"

        # compare_clustering_scores(
        #     **self._clustering_scores_kwargs(),
        #     dataset_type=dataset_type,
        #     models2process_dict=model_configs_dict
        # )
        try:
            res = compare_clustering_scores(
                **self._clustering_scores_kwargs(),
                dataset_type=dataset_type,
                models2process_dict=model_configs_dict,
            )
            return np.round(res,2)
        except (KeyError, FileNotFoundError) as exc:
            raise RuntimeError(
                "Default config strategy failed. "
                "Either (i) compute your AE run with `get_pca=True`, or "
                "(ii) provide a valid `configs.json` next to every trained model."
            ) from exc
    


    def load_configs(self, path:str, config_filename:str="configs.json"):
        fp = os.path.join(path, config_filename)
        
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"config.json not found: {fp}")
        with open(fp, "r") as f:
            configs = json.load(f)
        return configs


    def _get_pca_from_configs(self, configspath:str) -> str:
        configs = self.load_configs(configspath)
        pca_val = configs.get("get_pca")
        if pca_val == True:
            return "preprocess_results_model_pca_format"
        return "process_single_model_format"

    def genomap(self,
        model_result_folder_dict:Optional[Dict[str,str]]=None,
        models:Optional[List[str]]=None, # Default is to do all models in model_result_folder_dict.
        types:Optional[List[str]]=None,  # ["train", "test", "val"]
        splits:Optional[List[int]]=None, # i.e. [1, 2] ...
        celltype:Optional[List[str]]=None,
        batches:Optional[List[str]]=None,
        n_cells_per_batch:int=None,
        n_batches:int=None,
        n_genes:int=None,
        n_col:int=None,
        n_row:int=None,
        gene_index_col:str=None,
        scaling:str=None,
        add_inputs_fe:bool=None, 
        extra_recon:str=None, # "fe" or "all" ( Not really sure what this does.)
        seed:int=None, # random seed
        n_cells_2_plot: int = None,
        n_top_genes : int = None,
        min_val : int = None,
        max_val : int = None,
        num_iter: int = None ,
        cell_id_col : str = None,
        issparse : bool = None,
        ):
        # This will update self.paths which is the basis for _genomap_kwargs
        # Not ideal, but workable.
        self._check_update_assertions(model_result_folder_dict)
        
                # 1. Build GenomapConfig from core paths + tuning knobs
        core = self._genomap_kwargs()
        core['celltype'] = celltype if celltype is not None else core['celltype']
        core['n_cells_per_batch'] = n_cells_per_batch if n_cells_per_batch is not None else core['n_cells_per_batch']
        core['batches'] = batches if batches is not None else core['batches']
        core['n_batches'] = n_batches if n_batches is not None else core['n_batches']
        core['n_genes'] = n_genes if n_genes is not None else core['n_genes']
        core['n_col'] = n_col if n_col is not None else core['n_col']
        core['n_row'] = n_row if n_row is not None else core['n_row']
        core['gene_index_col'] = gene_index_col if gene_index_col is not None else core['gene_index_col']
        core['scaling'] = scaling if scaling is not None else core['scaling']
        core['add_inputs_fe'] = add_inputs_fe if add_inputs_fe is not None else core['add_inputs_fe']
        core['extra_recon'] = extra_recon if extra_recon is not None else core['extra_recon']
        core['seed'] = seed if seed is not None else core['seed']
        core['n_cells_2_plot'] = n_cells_2_plot if n_cells_2_plot is not None else core['n_cells_2_plot']
        core['n_top_genes'] = n_top_genes if n_top_genes is not None else core['n_top_genes']
        core['min_val'] = min_val if min_val is not None else core['min_val']
        core['max_val'] = max_val if max_val is not None else core['max_val']
        core['num_iter'] = num_iter if num_iter is not None else core['num_iter']
        core['cell_id_col'] = cell_id_col if cell_id_col is not None else core['cell_id_col']
        core['issparse'] = issparse if issparse is not None else core['issparse']

        cfg = GenomapConfig(
            compare_models_path = core["compare_models_path"],
            data_base_path      = core["data_base_path"],
            scenario_id         = core["scenario_id"],
            input_base_path     = core["input_base_path"],
            analysis_name       = core["analysis_name"],
            celltype            = core['celltype'],
            batches             = core["batches"],
            n_cells_per_batch   = core["n_cells_per_batch"],
            n_batches           = core["n_batches"],
            n_genes             = core["n_genes"],
            n_col               = core["n_col"],
            n_row               = core["n_row"],
            num_iter            = core["num_iter"],
            cell_id_col         = core["cell_id_col"],
            gene_index_col      = core["gene_index_col"],
            scaling             = core["scaling"],
            add_inputs_fe       = core["add_inputs_fe"],
            extra_recon         = core["extra_recon"],
            seed                = core["seed"],
            issparse            = core["issparse"],
            n_cells_2_plot      = core["n_cells_2_plot"],
            n_top_genes         = core["n_top_genes"],
            min_val             = core["min_val"],
            max_val             = core["max_val"],
        )

        if models is None:
            models = list(model_result_folder_dict.keys())

        # 2. Delegate to legacy wrapper that expects (run_names_dict, results_path_dict, cfg, ?)
        return genomap_and_plot(
            core["run_names_dict"],
            core["results_path_dict"],
            cfg,
            models=models,
            types=types,
            splits=splits,
        )

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
    def __init__(self, model_result_folder_dict:Optional[Dict[str,str]]=None, analysis_name:Optional[str]=None,
                  experiment_name:str="AML", *args, **kwargs) -> None:
        super().__init__(experiment_name=experiment_name, model_result_folder_dict=model_result_folder_dict, analysis_name=analysis_name)
        self.paths = self.analysis_named_experiment_paths(experiment_name, model_result_folder_dict, analysis_name)

    def _clustering_scores_kwargs(self) -> Dict[str,str]:
        return {
            "run_names_dict":self.paths.model_results_folder_dict,
            "results_path_dict":self.paths.latent_space_path,
            "compare_models_path":self.paths.outputs_path,
            "analysis_name":self.paths.analysis_name
        }

    def _genomap_kwargs(self):
        return {
            "run_names_dict":self.paths.model_results_folder_dict,
            "results_path_dict":self.paths.latent_space_path,
            "compare_models_path":self.paths.outputs_path,
            "data_base_path":self.paths.data_path,
            "scenario_id":self.paths.data_folder_name,
            "input_base_path":self.paths.splits_path,
            "analysis_name":self.paths.analysis_name,
            "n_cells_per_batch":300,
            "n_batches":19,
            "n_genes":2916,
            "n_col":54,
            "n_row":54,
            "gene_index_col":"Gene",
            "scaling":"min_max",
            "add_inputs_fe":True,
            "extra_recon":"fe",
            "seed":42,
            "issparse":False,
            "n_cells_2_plot":4,
            "n_top_genes":10,
            "num_iter":100,
            "celltype":["Mono", "Mono-like"],
            "batches":["AML420B", "BM5", "MUTZ3"],
            "cell_id_col":"Cell",
            "min_val":-1, 
            "max_val":2,
        }

    
    def _umap_kwargs(self):
        return {
            "run_names_dict":self.paths.model_results_folder_dict,
            "results_path_dict":self.paths.latent_space_path,
            "compare_models_path":self.paths.outputs_path,
            "input_base_path":self.paths.splits_path,
            "analysis_name":self.paths.analysis_name,
            "extra_color_cols":["Patient_group"],
            "issparse":False,
        }


class ASDAnalysis(Analysis):
    def __init__(self, model_result_folder_dict:Optional[Dict[str,str]]=None, analysis_name:Optional[str]=None,
                  experiment_name:str="ASD", *args, **kwargs) -> None:
        super().__init__(experiment_name=experiment_name, model_result_folder_dict=model_result_folder_dict, analysis_name=analysis_name)
        self.paths = self.analysis_named_experiment_paths(experiment_name, model_result_folder_dict, analysis_name)

    def _clustering_scores_kwargs(self) -> Dict[str,str]:
        return {
            "run_names_dict":self.paths.model_results_folder_dict,
            "results_path_dict":self.paths.latent_space_path,
            "compare_models_path":self.paths.outputs_path,
            "analysis_name":self.paths.analysis_name
        }

    def _genomap_kwargs(self):
        return{
            "run_names_dict":self.paths.model_results_folder_dict,
            "results_path_dict":self.paths.latent_space_path,
            "compare_models_path":self.paths.outputs_path,
            "data_base_path":self.paths.data_path,
            "scenario_id":self.paths.data_folder_name,
            "input_base_path":self.paths.splits_path,
            "analysis_name":self.paths.analysis_name,
            "n_cells_per_batch":300,
            "n_genes":2916,
            "n_col":54,
            "n_row":54,
            "scaling":"min_max",
            "add_inputs_fe":True,
            "extra_recon":"fe",
            "seed":42,
            "issparse":False,
            "n_top_genes":10,
            "num_iter":100,
            # These changed
            "n_batches":31,
            "cell_id_col":"cell",
            "gene_index_col":"gene_ids",
            "celltype":["L2/3"],
            "batches": [
                "donor_5531",
                "donor_5945",
                "donor_5419",
                "donor_6032",
                "donor_5242",
                "donor_5976",
            ],
            "n_cells_2_plot":6,
            "min_val":-2, 
            "max_val":6,
        }

    def _umap_kwargs(self):
        return {
            "run_names_dict":self.paths.model_results_folder_dict,
            "results_path_dict":self.paths.latent_space_path,
            "compare_models_path":self.paths.outputs_path,
            "input_base_path":self.paths.splits_path,
            "analysis_name":self.paths.analysis_name,
            "n_batches":31,
            "n_neighbors":15,
            "rng_seed":5,
            "scaling":"min_max",
            "batch_col":"batch",
            "shape_col":"celltype",
            "color_col":"celltype",
            "use_rep":"X_umap",
            "extra_color_cols":["diagnosis"],
            "issparse":False,
        }



class HHAnalysis(Analysis):
    def __init__(self, model_result_folder_dict:Optional[Dict[str,str]]=None, analysis_name:Optional[str]=None,
                  experiment_name:str="HH", *args, **kwargs) -> None:
        super().__init__(experiment_name=experiment_name, model_result_folder_dict=model_result_folder_dict, analysis_name=analysis_name)
        self.paths = self.analysis_named_experiment_paths(experiment_name, model_result_folder_dict, analysis_name)

    def _clustering_scores_kwargs(self) -> Dict[str,str]:
        return {
            "run_names_dict":self.paths.model_results_folder_dict,
            "results_path_dict":self.paths.latent_space_path,
            "compare_models_path":self.paths.outputs_path,
            "analysis_name":self.paths.analysis_name
        }

    def _genomap_kwargs(self):
        return{
            "run_names_dict":self.paths.model_results_folder_dict,
            "results_path_dict":self.paths.latent_space_path,
            "compare_models_path":self.paths.outputs_path,
            "data_base_path":self.paths.data_path,
            "scenario_id":self.paths.data_folder_name,
            "input_base_path":self.paths.splits_path,
            "analysis_name":self.paths.analysis_name,
            "n_cells_per_batch":300,
            "n_genes":2916,
            "n_col":54,
            "n_row":54,
            "scaling":"min_max",
            "add_inputs_fe":True,
            "extra_recon":"fe",
            "seed":42,
            "issparse":True,
            "n_top_genes":10,
            "num_iter":100,
            # These changed
            "n_batches":31,
            "cell_id_col":"_index",
            "gene_index_col":"_index",
            "celltype":["Ventricular_Cardiomyocyte", "Endothelial", "Fibroblast", "Pericytes"],
            "batches": ["H0037_Apex", "HCAHeart7836681", "HCAHeart8102861", "H0015_septum"],
            "n_cells_2_plot":4,
            "min_val":-2, 
            "max_val":8,
        }

    def _umap_kwargs(self):
        return {
            "run_names_dict":self.paths.model_results_folder_dict,
            "results_path_dict":self.paths.latent_space_path,
            "compare_models_path":self.paths.outputs_path,
            "input_base_path":self.paths.splits_path,
            "analysis_name":self.paths.analysis_name,
            "extra_color_cols":["DonorID","TissueDetail","protocol"],
            "issparse":True,
        }
