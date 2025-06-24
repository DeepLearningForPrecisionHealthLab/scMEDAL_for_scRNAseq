""" All paths/directories assume CWD is the git repository"""
import os
DATA_DIR = os.path.join(os.getcwd(), "data")
OUTPUTS_DIR = os.path.join(os.getcwd(), "outputs")


## AML
AML_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "AML")
AML_DATA_DIR = os.path.join(DATA_DIR, "AML_data")
AML_DATA_PATH = os.path.join(DATA_DIR, "AML", "AML_data.zip")
AML_EXPERIMENT_NAME = "log_transformed_2916hvggenes"


os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(AML_OUTPUTS_DIR, exist_ok=True)
