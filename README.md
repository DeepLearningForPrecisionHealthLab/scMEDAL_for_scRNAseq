# ARMED_genomics



## Welcome to ARMED Genomics repository

## Description
The goal of this project is to obtain meaningful latent representations from single cell RNA seq data. To obtain this latent space, we will take inspiration from Nguyen et al 2023 and apply an ARMED vector autoencoder to a single cell RNA seq data gene expression matrix, also known as count matrix. 

For the simulation examples, the fixed effects are the celltypes and the random effects are the donors. The random effects are known in the literature as batch effects.

* The inputs are count matrices of size = n cells * m genes.
* The outputs are reconstructed count matrices of the same size than the input.
* The latent space is of size = n cells * p reduced features

## ARMED Framework Overview

The ARMED framework integrates seamlessly with an Autoencoder Classifier (AEC) to process gene expression count matrices (\(X\)) and predict cell types (\(\hat{y}\)). This framework is designed to enhance the conventional AEC by addressing both predictable and variable batch effects through its dual subnetwork architecture.

### Subnetworks Description

#### Fixed Effects Subnetwork

![ARMED Fixed Effects Subnetwork](./images/FEsubnet.png)

The Fixed Effects Subnetwork augments the AEC with an adversarial classifier. This configuration aims to refine the model by preventing the prediction of predictable batch effects, thereby improving the accuracy and reliability of the AEC in standardized conditions.

#### Random Effects Subnetwork

![ARMED Random Effects Subnetwork](./images/REsubnet.png)

The Random Effects Subnetwork extends the AEC by incorporating a design matrix for batch effects into each layer of an autoencoder. It utilizes variational layers where the weight distributions (\(p\)) are optimized to approximate a target distribution (\(q\)), addressing inter-cluster variability effectively. Additionally, this subnetwork includes a classifier specifically designed to ensure the predictability and consistency of batch effects across different datasets.

### Usage

To implement the ARMED framework in your research or applications, ensure that your dataset includes a well-defined gene expression count matrix and corresponding cell type annotations. By leveraging the ARMED model, researchers can achieve more nuanced and robust analyses, crucial for studies involving complex biological datasets. 

For further details on model architecture and implementation, refer to the diagrams provided for each subnetwork. These visual aids will help clarify the operational dynamics and the strategic enhancements made to the traditional AEC model.
![ARMED Subnetworks](./images/latent_spaces.png)


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

# Experiment files

* [heart_data](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data): Experiment with the Healthy Human Heart dataset retrieved from [here](https://figshare.com/articles/dataset/Batch_Alignment_of_single-cell_transcriptomics_data_using_Deep_Metric_Learning/20499630/2) (Yu et al 2023)
    * [run_models](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data/run_models)
        * [AEC](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/AEC)
        * [AEC_DA](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/AEC_DA)
        * [AE_RE](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/AE_RE)

    * [5fold_cross_val](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data/preprocessing/5fold_cross_val)



### Experiment Structure

* **model_config.py**
  * Used to update model configurations and set the output path.

* **Scripts**
  * **run_modelname_allfolds.py**
    * Executes the model across 5 folds.
    * Command to run the script: `python run_modelname_allfolds.py`
  * **sbatch_run_modelname.sh**
    * Shell scripts to be submitted to Slurm using: `sbatch yourscript.sh`



### Environment
The experiments are conducted using the ARMED_Aixa_v2 environment.