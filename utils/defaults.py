""" All paths/directories assume CWD is the git repository"""
import os
DEFAULTS_ABS_PATH = os.path.abspath(__file__)
DATA_DIR = os.path.join(os.getcwd(), "data")
OUTPUTS_DIR = os.path.join(os.getcwd(), "outputs")

AML_MODEL_KWARGS = {"n_clusters":19, "n_pred":21}
ASD_MODEL_KWARGS = {"n_clusters":31, "n_pred":17}
HH_MODEL_KWARGS = {"n_clusters":147, "n_pred":13}

#### AML
AML_DATA_PATH = f"/archive/bioinformatics/DLLab/AixaAndrade/src/gitfront/scMEDAL_for_scRNAseq/Experiments/data/AML_data/",
AML_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "AML")
AML_DATA_DIR = os.path.join(DATA_DIR, "AML_data")
#AML_DATA_PATH = os.path.join(DATA_DIR, "AML", "AML_data.zip")
AML_EXPERIMENT_NAME = "log_transformed_2916hvggenes"


os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(AML_OUTPUTS_DIR, exist_ok=True)
