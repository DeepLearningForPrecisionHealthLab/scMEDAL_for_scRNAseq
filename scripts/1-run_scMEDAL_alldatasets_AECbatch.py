import sys, os
# Set up your project path here
#os.chdir("/archive/bioinformatics/DLLab/AixaAndrade/src/gitfront/batchclassifiertest/scMEDAL_for_scRNAseq")
sys.path.insert(0, os.getcwd())          # <-- add this right after the chdir

# The working dir should be scMEDAL_for_scRNAseq dir
print("working dir:",os.getcwd())
from models.models import train_model_on_named_experiment

# Note: The default AEC configuration is set up for AEC_ct (predicting cell-type labels).
# Below, we modify the configuration parameters so that AEC predicts batch labels (AEC_batch).

# AML
aec_aml = train_model_on_named_experiment(
  "AEC", "AML",
  model_kwargs={
    "n_latent_dims": 50,
    "bio_col": "batch",   # makes classifier target be batch
    "get_pred": True,     # ensures outputs = (x, onehot(bio_col))
    "use_z": False,       # AEC input stays x
    # override n_pred to classify batches
    "n_pred": 19,
  },
)

# ASD
aec_asd = train_model_on_named_experiment(
  "AEC", "ASD",
  model_kwargs={
    "n_latent_dims": 50,
    "bio_col": "batch",   # makes classifier target be batch
    "get_pred": True,     # ensures outputs = (x, onehot(bio_col))
    "use_z": False,       # AEC input stays x
    # override n_pred to classify batches
    "n_pred": 31,
  },
)


# HH
aec_hh = train_model_on_named_experiment(
  "AEC", "HH",
  model_kwargs={
    "n_latent_dims": 50,
    "bio_col": "batch",   # makes classifier target be batch 
    "get_pred": True,     # ensures outputs = (x, onehot(bio_col))
    "use_z": False,       # AEC input stays x
    # override n_pred to classify batches
    "n_pred": 147,
  },
)
