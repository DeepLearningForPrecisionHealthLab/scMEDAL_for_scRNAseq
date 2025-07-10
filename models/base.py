import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json
import glob
import configs.configs as cfg
import tensorflow as tf
import pandas as pd
import numpy as np
from abc import ABC
from typing import Optional, Dict, Any
from types import SimpleNamespace
from utils.model_train_utils import generate_run_name, run_all_folds



class Model(ABC):
    valid_models=["ae","aec","scmedalfe","scmedalfec", "scmedalre", "saucie", "mec"]
    valid_named_experiment=["AML","ASD", "HH"]

    def __init__(self, model_name:str, **kwargs):
        assert model_name in self.valid_models, f"Invalid model {model_name}. Valid models include: {self.valid_models}"
        self.model_name = model_name
        self.compile_configs:cfg.CompileConfigs=self._init_compile_configs(kwargs)
        self.model_configs:cfg.ModelConfigs=self._init_model_configs(kwargs)
        self.data_configs:cfg.DataConfigs=self._init_data_configs(kwargs)
        self.training_configs:cfg.TrainingConfig=self._init_training_configs(kwargs)
        self.scores_configs:cfg.ScoreConfig=self._init_score_configs(kwargs)
        self.expt_design_configs:cfg.ExperimentDesignConfigs=self._init_exp_design_configs(kwargs)
        self.model_params:Dict[str,Any] = self.__init_modelparams()
        self.latent_key = None

    def _init_compile_configs(self, kwargs:Optional[Dict[str, Any]]=None):
        self.compile_configs = cfg.CompileConfigs(self.model_name).configs        
        if kwargs is not None:
            self.compile_configs.update(**{k:v for k,v in kwargs.items() if k in self.compile_configs.keys()})
        return self.compile_configs

    def _init_model_configs(self, kwargs:Optional[Dict[str, Any]]=None):
        self.model_configs = cfg.ModelConfigs(self.model_name)        
        if kwargs is not None:
            self.model_configs = self.model_configs.configs._replace(**{k:v for k,v in kwargs.items() if k in self.model_configs.configs._fields})
        return self.model_configs
      
    def _init_data_configs(self, kwargs:Optional[Dict[str, Any]]=None):
        self.data_configs = cfg.DataConfigs(self.model_name).configs        
        if kwargs is not None:
            self.data_configs = self.data_configs._replace(**{k:v for k,v in kwargs.items() if k in self.data_configs._fields})
        return self.data_configs
    
    def _init_training_configs(self, kwargs:Optional[Dict[str, Any]]=None):
        self.training_configs = cfg.TrainingConfigs(self.model_name)        
        if kwargs is not None:
            self.training_configs = self.training_configs._replace(**{k:v for k,v in kwargs.items() if k in self.training_configs._fields})
        return self.training_configs

    def _init_score_configs(self, kwargs:Optional[Dict[str, Any]]=None):
        self.score_configs = cfg.ScoreConfigs(self.model_name).configs        
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
            **self.compile_configs,
            **self.model_configs._asdict(),
            **self.data_configs._asdict(),
            **self.training_configs._asdict(),
            **self.scores_configs._asdict(),
            **self.expt_design_configs._asdict(), 
        }
        ignore = self.model_params.get("ignore", [])
        self.model_params.pop("ignore")
        # self.latent_key = self.model_params.get("latent_keys_config")
        # if self.latent_key is not None:
        #     self.model_params.pop("latent_keys_config")
        self.model_params['run_name'] = generate_run_name(self.model_params, ignore, name="run_crossval") 
        return self.model_params

    def _is_jsonable(self, input):
        try:
            json.dumps(input)
            return True
        except:            
            return False        

    def save_configs(self, path:str) -> None:
        # Ugly, but necessary b/c currently implementation uses instantiated Loss, Metrics, and Optimizer objects 
        configs_json = {}
        for k, v in self.model_params.items():
            if self._is_jsonable(v):
                configs_json[k] = v
            else:
                configs_json[k] = [f"Unable To Json encode {v}", str(v)]
        
        with open(path, "w") as f:    
            json.dump(configs_json, f)
    
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


    def run_train(self, data_path:Optional[str]=None, outputs_path:Optional[str]=None, named_experiment:Optional[str]=None, save_model:bool=True, quick:bool=False, plotconfigs:Optional[cfg.PlotConfigs]=None, plot_kwargs:Optional[Dict[str, Any]]=None): 
        """
        Quick sets epochs to 3.
        """
        if plotconfigs is None:
            plotconfigs = cfg.PlotConfigs() if plot_kwargs is None else cfg.PlotConfigs(**plot_kwargs)
        shape_color_dict = plotconfigs.get_shape_color_dict(self.expt_design_configs)

        if quick:
            self.training_configs._replace(epochs=10)
            self.model_params['epochs'] = 10
            self.model_params['fold_list'] = [1]#[1,2]

        model_name = self.model_name

        if outputs_path is None:
            outputs_path = os.path.join(os.getcwd(), "outputs")

        if named_experiment is not None:
            paths = self.load_named_experiment_paths(named_experiment)
            data_path = paths.get("data_path")
            folder_name = paths.get("scenario_id")
            splits_path = os.path.join(data_path, folder_name, paths.get("splits_folder"))
            outputs_path = os.path.join(outputs_path, named_experiment)
        
        issparse, load_dense = False, False
        if named_experiment == "HH":
            issparse, load_dense = True, True

        print(f"Parent folder: {splits_path}")

        saved_models_base = os.path.join(outputs_path, "saved_models", folder_name, model_name)
        figures_base = os.path.join(outputs_path, "figures", folder_name, model_name)
        latent_space_base = os.path.join(outputs_path, "latent_space", folder_name, model_name)

        # --------------------------------------------------------------------------------------
        # 12. Define Base Paths Dictionary
        # --------------------------------------------------------------------------------------
        base_paths_dict = {
            "models": saved_models_base,
            "figures": figures_base,
            "latent": latent_space_base
        }

        print("Save model set to:", save_model)

        params=SimpleNamespace(**self.model_params)

        # --------------------------------------------------------------------------------------
        # 2. Load Metadata and Define Categories
        # --------------------------------------------------------------------------------------
        # Load metadata before splits
        metadata_all = pd.read_csv(glob.glob(os.path.join(data_path, folder_name) + "/*meta.csv")[0])

        # Convert columns to categorical types
        metadata_all['celltype'] = metadata_all['celltype'].astype('category')
        if "batch" in metadata_all.columns:
            metadata_all['batch'] = metadata_all['batch'].astype('category')
        if "sampleID" in metadata_all.columns:
            metadata_all['batch'] = metadata_all['sampleID'].astype('category')

        # Print the number of unique batches
        print("Number of batches:", len(np.unique(metadata_all[params.batch_col])))

        # Define One Hot Encoded (OHE) order for donor and celltype categories
        seen_donor_ids = np.unique(metadata_all[params.batch_col]).tolist()
        print("Ordered batches (donors):", seen_donor_ids)

        celltype_ids = np.unique(metadata_all[params.bio_col]).tolist()

        # --------------------------------------------------------------------------------------
        # 3. Run All Folds
        # --------------------------------------------------------------------------------------
        # Run folds and compute clustering metrics
        mean_scores = run_all_folds(
            Model=self.alg,
            input_base_path=splits_path,
            out_base_paths_dict=base_paths_dict,
            folds_list=params.fold_list,
            run_name=params.run_name,
            model_params_dict=self.model_params,
            build_model_dict={k:v for k, v in self.model_configs._asdict().items() if k != "ignore"},
            compile_dict=self.compile_configs,
            save_model=save_model,
            batch_col=params.batch_col,
            bio_col=params.bio_col,
            batch_col_categories=seen_donor_ids,
            bio_col_categories=celltype_ids,
            model_type=self.model_name,
            issparse=issparse,
            load_dense=load_dense,
            shape_color_dict=shape_color_dict,
            sample_size=params.sample_size
        )

        # --------------------------------------------------------------------------------------
        # 4. Save Configuration File
        # --------------------------------------------------------------------------------------
        destination_path = os.path.join(saved_models_base, params.run_name, "configs.json")
        self.save_configs(destination_path)
