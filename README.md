# ARMED_genomics



## Welcome to ARMED Genomics repository

## Description
The goal of this project is to obtain meaningful latent representations from single cell RNA seq data. To obtain this latent space, we will take inspiration from Nguyen et al 2023 and apply an ARMED vector autoencoder to a single cell RNA seq data gene expression matrix, also known as count matrix. 

For the simulation examples, the fixed effects are the celltypes and the random effects are the donors. The random effects are known in the literature as batch effects.

* The inputs are count matrices of size = n cells * m genes.
* The outputs are reconstructed count matrices of the same size than the input.
* The latent space is of size = n cells * p reduced features

![ARMED model](./images/MEDL.png)

# Main files

* models
    * [AE_v4.py](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/blob/main/models/AE_v4.py): AEC (simple autoencoder classifier), DA_AE (domain adversarial autoencoder for fixed effects), DomainEnhancingAutoencoderClassifier (Random Effects autoencoder) models
    * [random_effects.py](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/blob/main/models/random_effects.py): random effects classes (written by Kevin Nguyen for the original ARMED paper)
* utils
    * [utils.py](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/utils)

# Experiment files

* heart_data(https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data): Experiment with the Healthy Human Heart dataset retrieved from https://figshare.com/articles/dataset/Batch_Alignment_of_single-cell_transcriptomics_data_using_Deep_Metric_Learning/20499630/2 (Yu et al 2023)