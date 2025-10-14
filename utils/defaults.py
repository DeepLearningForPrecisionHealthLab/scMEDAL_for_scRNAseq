""" All paths/directories assume CWD is the git repository"""
import os
DEFAULTS_ABS_PATH = os.path.abspath(__file__)

# REPOSITORY PATH
# update your ROOT_PATH
ROOT_PATH = "/archive/bioinformatics/DLLab/AixaAndrade/src/gitfront/dev2/scMEDAL_for_scRNAseq"
DATA_DIR = os.path.join(ROOT_PATH,"data") 
OUTPUTS_DIR =  os.path.join(ROOT_PATH,"outputs")
print("OUTPUTS_DIR:",OUTPUTS_DIR)


AML_MODEL_KWARGS = {"n_clusters":19, "n_pred":21}
ASD_MODEL_KWARGS = {"n_clusters":31, "n_pred":17}
HH_MODEL_KWARGS = {"n_clusters":147, "n_pred":13}


ASD_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "ASD")
HH_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "HH")
AML_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "AML")


os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(AML_OUTPUTS_DIR, exist_ok=True)
os.makedirs(ASD_OUTPUTS_DIR, exist_ok=True)
os.makedirs(HH_OUTPUTS_DIR, exist_ok=True)


AML_DATA_DIR = os.path.join(DATA_DIR, "AML_data")
AML_EXPERIMENT_NAME = "log_transformed_2916hvggenes"

ASD_DATA_DIR = os.path.join(DATA_DIR, "ASD_data")
ASD_EXPERIMENT_NAME = "log_transformed_2916hvggenes"


HH_DATA_DIR = os.path.join(DATA_DIR, "HH_data")
HH_EXPERIMENT_NAME = "log_transformed_3000hvggenes"



AML_PATHS_CONFIG = {
    "data_base_path":AML_DATA_DIR,
    "scenario_id":AML_EXPERIMENT_NAME,
    "outputs_path": AML_OUTPUTS_DIR,
}

ASD_PATHS_CONFIG =  {
    "data_base_path": ASD_DATA_DIR,
    "scenario_id":ASD_EXPERIMENT_NAME,
    "outputs_path": ASD_OUTPUTS_DIR,
}

HH_PATHS_CONFIG = {
    "data_base_path": HH_DATA_DIR,
    "scenario_id":HH_EXPERIMENT_NAME,
    "outputs_path": HH_OUTPUTS_DIR,
}