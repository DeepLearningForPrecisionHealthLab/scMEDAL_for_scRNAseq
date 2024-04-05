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

# Main files

* models
    * [AE_v4.py](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/blob/main/models/AE_v4.py): AEC (simple autoencoder classifier), DA_AE (domain adversarial autoencoder for fixed effects), DomainEnhancingAutoencoderClassifier (Random Effects autoencoder) models
    * [random_effects.py](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/blob/main/models/random_effects.py): random effects classes (written by Kevin Nguyen for the original ARMED paper)
* utils
    * [utils.py](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/utils)

# Experiment files

* [heart_data](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data): Experiment with the Healthy Human Heart dataset retrieved from [here](https://figshare.com/articles/dataset/Batch_Alignment_of_single-cell_transcriptomics_data_using_Deep_Metric_Learning/20499630/2) (Yu et al 2023)