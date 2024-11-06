# Mixed Effects Deep Learning (MEDL) Autoencoder for Interpretable Analysis of Single-cell RNA Sequencing (scRNA-seq) Data

Updated README description and healthy heart experiment on 11/6/2024

## Description

The Mixed Effects Deep Learning (MEDL) framework is designed to extract meaningful latent representations from single-cell RNA sequencing (scRNA-seq) data while accounting for batch effects?common confounders that can obscure true biological signals. By extending linear mixed-effects models into a nonlinear context, MEDL effectively addresses the inherent non-linearity in confounded scRNA-seq datasets.

**Key Features:**

- **Inputs:** Gene expression count matrix \( X \in \mathbb{R}^{n \times m} \), where \( n \) is the number of cells and \( m \) is the number of genes.
- **Outputs:** Reconstructed gene expression matrix \( \hat{X} \) and latent space representations.
- **Latent Space:** Reduced feature space of size \( n \times p \), capturing essential biological variability.

## MEDL Framework Overview

The MEDL framework consists of two parallel subnetworks that jointly model fixed and random effects within gene expression data:

1. **Fixed Effects Subnetwork (MEDL-AE-FE):** Captures batch-invariant features by suppressing batch effects. It employs an autoencoder with weight tying, dense hidden layers, and an adversarial classifier that learns to predict batch labels. The loss function balances reconstruction error and adversarial loss to mitigate batch-specific variations.

2. **Random Effects Subnetwork:** Models batch-specific variations using variational inference. It approximates true batch distributions with optimized surrogate posteriors and includes a classifier for batch label prediction. By maximizing the Evidence Lower Bound (ELBO), it ensures the latent space encodes batch-specific information while regularizing the model to prevent overfitting.

![MEDL Diagram](./images/MEDL.png)

### Subnetwork Details

- **Fixed Effects Loss Function:**
  \[
  L_{\text{FE}} = \lambda_{\text{MSE}} \cdot L_{\text{MSE}}(X, \hat{X}) - \lambda_{\text{A}} \cdot L_{\text{CCE}}(z, \hat{z})
  \]
  Balances reconstruction and adversarial losses to capture fixed effects.

- **Random Effects Loss Function:**
  \[
  L_{\text{RE}} = \lambda_{\text{MSE}} \cdot L_{\text{MSE}}(X, \hat{X}') + \lambda_{\text{CCE}} \cdot L_{\text{CCE}}(z, \hat{z}') + \lambda_{\text{KL}} \cdot D_{\text{KL}}(q(U) \| p(U))
  \]
  Combines reconstruction error, batch classification loss, and KL divergence for modeling random effects.

## Usage

To implement the MEDL framework:

1. **Data Preparation:** Ensure your dataset includes a gene expression count matrix and corresponding batch labels (e.g., donor IDs, experimental batches).

2. **Model Training:** Use the provided MEDL-AE code to train on your data. The model outputs reconstructed gene expression matrices and latent space representations with reduced batch effects.

3. **Downstream Analysis:** Utilize the latent representations for tasks like clustering, visualization, or differential expression analysis to uncover true biological signals.



By leveraging the MEDL framework, researchers can achieve more accurate and interpretable analyses of scRNA-seq data, effectively separating true biological variability from technical noise due to batch effects.

### Models
* [AE_v4.py](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/blob/main/models/AE_v4.py): Contains models including the simple AEC (Autoencoder Classifier), DA_AE (Domain Adversarial Autoencoder for fixed effects), and the DomainEnhancingAutoencoderClassifier (for Random Effects).
* [random_effects.py](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/blob/main/models/random_effects.py): Implements random effects classes, originally developed by Kevin Nguyen for the ARMED paper.

### Utilities
* [utils.py](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/blob/main/utils/utils.py)
  * Functions for reading and saving data, plotting, and calculating clustering scores.
* [model_train_utils.py](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/blob/main/utils/model_train_utils.py)
  * Functions to load data, build, and train models using configuration information.
* [splitter.py](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/blob/main/utils/splitter.py)
  * Function to split data into training, validation, and test sets. Includes support for 5-fold cross-validation.
* [utils_load_model.py](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/blob/main/utils/utils_load_model.py)
  * Functions to load previously saved models.

# Experiment Files

## Healthy Heart Data Experiment
Access the Healthy Human Heart dataset utilized in this experiment from [here](https://figshare.com/articles/dataset/Batch_Alignment_of_single-cell_transcriptomics_data_using_Deep_Metric_Learning/20499630/2) (Yu et al., 2023).

### Models
Explore the models used in the Heart Data Experiment:
- **[Run Models Directory](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data/run_models)**
  - [Autoencoder (AE)](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/AE)
  - [Autoencoder Classifier (AEC)](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/AEC)
  - [Fixed Effects Subnetwork (AEC_DA)](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/AEC_DA)
  - [Fixed Effects Subnetwork (AE_DA)](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/AE_DA): This model does not have a classifier.
  - [Random Effects Subnetwork (AE_RE)](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/AE_RE)

### 5-Fold Cross-Validation
- **[5-Fold Cross-Validation Directory](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data/preprocessing/5fold_cross_val)**
  - [Create Splits Notebook](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/blob/main/heart_data/preprocessing/5fold_cross_val/create_splits.ipynb): A notebook for splitting data using 5-fold cross-validation.
  - [Config Split Paths Script](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/blob/main/heart_data/preprocessing/5fold_cross_val/config_split_paths.py): Manages paths for input data and data splits.
  - [Check Splits Notebook](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/blob/main/heart_data/preprocessing/5fold_cross_val/check_splits.ipynb): Ensures no data leakage between training, testing, and validation sets.


# Experiment Configuration

### Model Configuration Script

- **Model Configuration**
  - Each model has its own `model_config.py` file. For an example, see the [AEC model configuration](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/blob/main/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/AEC/model_config.py).
  - This script updates model settings and defines output paths.

Make sure you update your data paths in `model_config.py`:

```python
data_base_path = "path to base path with all variants of the experiment"
scenario_id = "path to specific pre-processing of your experiment"
```

## Construct the Data Paths

- **Path to the data from the experiment you want to run:**

```python
data_path = os.path.join(data_base_path, scenario_id)
```

- **Path to the folder that contains the specific splits (for k-fold cross validation you are running):**

```python
data_seen = os.path.join(data_base_path, scenario_id, 'splits')
print(f"Parent folder: {data_seen}")
```

## Base Output Path

Update the following variable in your script:

```python
outputs_path = "/path/to/outputs"
```

## Folder and Model Naming

Define how you want to name your experiment and the model folder name:

```python
folder_name = "how you want to name your expt"
model_name = "AE_RE"  # the folder name of your output
```

# Script Execution

- **Run Model Across All Folds**
  - Command: `python run_modelname_allfolds.py`
- **Slurm Submission Script**
  - Command: `sbatch sbatch_run_modelname.sh`

### Execution Environment

- The experiments are executed within the ARMED_Aixa_v2 environment.

### Notes

Make sure to change the path to the utils folder in each of the files of the type: `run_modelname_allfolds.py`.
```


