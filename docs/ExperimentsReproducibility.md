# Experiment Reproducibility Guide

This guide maps each section of the paper to the corresponding **code**, **datasets**, and **model outputs**, so you can reproduce the results as described.

We conducted experiments on three datasets: **Healthy Heart (HH)**, **Autism Spectrum Disorder (ASD)**, and **Acute Myeloid Leukemia (AML)**. For each dataset, we provide:

* **1. [Datasets]**(../README.md#datasets)
  - Preprocessing and 5-fold cross-validation scripts to split the data into train/val/test.
  - Alternatively, we provide the **precomputed data splits** via Figshare. See **[Datasets](../README.md#datasets)**.
* **2. Scripts for model training** for our API models (Autoencoder, **scMEDAL** variants, [**Mixed Effects Classifier**](#mec-classifier-scripts)) and [**comparable models**](#comparable-models-scripts): **scVI**, **scANVI**, **Harmony**, **Scanorama**, and **SAUCIE**.
* [**3. Analysis notebooks**](#analysis-notebooks): Once you have trained your models you need to run the analysis notebooks that retrieve clustering scores (Figures 2-4 and 8), Genomaps (Figures 5-7), UMAPs (Figures 2-4 and 8) and extract the Random Forest classifier outputs reported in Table 1. Use these notebooks to reproduce the paper results once latent spaces and reconstruction outputs are available.

We also provide in the Figshare:

* **Latent space outputs**  and a **sample count matrix**  with 300-cell projections so you can reproduce the **Genomaps** exactly as in the paper.

You have two options:

## A. **Reproduce analyses from provided model outputs**
   Download our latent spaces and the 300-cell sample matrix, place them in the specified folders (see below), and run the analysis notebooks to regenerate paper figures.

      1. Place the AML latent space under:

          
          /outputs/AML/latent_space/log_transformed_2916hvggenes
          
      2. Place the 300-cell sample matrix used for Genomaps under:

        
          outputs/AML/compare_models/log_transformed_2916hvggenes/AML_default/genomap/CMmultibatch_300_cells_per_batch_19batches_Mono_Mono-like_with_2fe_input
        

> After these paths are set, run the analysis notebooks to reproduce the Genomaps section for AML.

## B. **Run models from scratch**
   Train the models yourself and then run the analysis notebooks.

### scMEDAL API Model scripts
To run all models from our API (AE,AEC,scMEDAL-FE,scMEDAL-RE and scMEDAL-FEC), use [1-run_scMEDAL_alldatasets.py](../scripts/1-run_scMEDAL_alldatasets.py) and make sure your **scMEDAL** environment is activated. See **[scMEDAL Installation](../README.md#Installation)**.

```bash
python 1-run_scMEDAL_alldatasets.py
```


If you train models yourself, you may need to update paths to match your own output directories. See [Outputs and analysis folders](../README.md#outputs-and-analysis-folders)


### MEC Classifier Scripts

After running the representation models, you can train the **Mixed Effects Classifier (MEC)**:

**ASD**

* [2-mec_asd_dx.py](../scripts/2-mec_asd_dx.py)
* [2-mec_asd_celltype.py](../scripts/2-mec_asd_celltype.py)

**AML**

* [2-mec_aml_celltype.py](../scripts/2-mec_aml_celltype.py)
* [2-mec_aml_patientgroup.py](../scripts/2-mec_aml_patientgroup.py)

**Healthy Heart**

* [2-mec_hh_tissue.py](../scripts/2-mec_hh_tissue.py)
* [2-mec_hh_celltype.py](../scripts/2-mec_hh_celltype.py)

---
### Comparable Models Scripts

Each comparable model has its own script and environment requirements. Configurations vary by dataset. Below we list the **AML** entry points; for **HH** and **ASD**, use the corresponding dataset paths/scripts.

> All comparable-model scripts produce the **same output file structure** as our API models. See **[Outputs & analysis folders](../README.md#outputs-and-analysis-folders)**.
##### SAUCIE

* Example Script (AML): [`comparables/AML/run_SAUCIE.py`](../comparables/AML/run_SAUCIE.py)
  
  Run:

  ```bash
  python run_SAUCIE.py
  ```
- See [requirements.txt](../comparables/SAUCIE/requirements.txt)
- Use [`SAUCIE.yaml`](../comparables/comparables_env/SAUCIE.yaml) tp create conda environment.

##### scVI

* Example Script (AML): [`comparables/AML/run_scVI.py`](../comparables/AML/run_scVI.py)
  
  Run:

  ```bash
  python run_scVI.py
  ```
- **Docs:** [https://scvi-tools.org/](https://scvi-tools.org/)


##### scanorama

* Example Script (AML): [`comparables/AML/run_scanorama.py`](../comparables/AML/run_scanorama.py)
  
  Run:

  ```bash
  python run_scanorama.py
  ```
- **Docs:** [https://scanpy.readthedocs.io/en/stable/external/preprocessing.html](https://scanpy.readthedocs.io/en/stable/external/preprocessing.html)


##### harmony

* Example Script (AML): [`comparables/run_harmony.py`](../comparables/AML/run_harmony.py)
  
  Run:

  ```bash
  python run_harmony.py
  ```
- **Docs:** [https://scanpy.readthedocs.io/en/stable/external/preprocessing.html](https://scanpy.readthedocs.io/en/stable/external/preprocessing.html)


##### scANVI

* Example Script (AML) : [`comparables/run_scANVI.py`](../comparables/AML/run_scANVI.py)
  
  Run:

  ```bash
  python run_scANVI.py
  ```
- **Docs:** [https://scvi-tools.org/](https://scvi-tools.org/)

Note:  Use **Comparables env for scVI, scANVI, harmony, scanorama and SAUCIE, use [comparables_env.yaml](../comparables/comparables_env/comparables_env.yaml) to create conda environment.


---


## Analysis Notebooks

Use the following notebooks to reproduce the analyses and figures:

* [3-analysis_aml.ipynb](../scripts/3-analysis_aml.ipynb): (Figure 4,Figure 7, Figure 8, Table 1)
* [3-analysis_asd.ipynb](../scripts/3-analysis_asd.ipynb): (Figure 3, Figure 6, Figure 8 and  Table 1)
* [3-analysis_hh.ipynb](../scripts/3-analysis_hh.ipynb): (Figure 2, Figure 5, Figure 8 and Table 1)

> Download/setup the required model outputs (or train from scratch), update paths where necessary, and run these notebooks to regenerate the paper results.



**Note:** Due to variability in TensorFlow, model outputs may differ slightly across runs. To account for this, we report 95% confidence intervals (CI) as an estimate of variability.



---

## References

- Litvinukova, M. et al. Cells of the adult human heart. Nature 588, 466-472 (2020).
- van Galen, P. et al. *Single-Cell RNA-Seq Reveals AML Hierarchies Relevant to Disease Progression and Immunity.* Cell 176, 1265-1281.e24 (2019).
- Velmeshev, D. et al. *Single-cell genomics identifies cell type-specific molecular changes in autism.* Science 364, 685-689 (2019).
- Speir, M. L. et al. *UCSC Cell Browser: visualize your single-cell data.* Bioinformatics 37, 4578-4580 (2021).
- Yu, X., Xu, X., Zhang, J., & Li, X. *Batch alignment of single-cell transcriptomics data using deep metric learning.* Nat Commun 14, 960 (2023).  
- Yu, X., Xu, X., Zhang, J., & Li, X. *Batch alignment of single-cell transcriptomics data using deep metric learning.* figshare [https://doi.org/10.6084/m9.figshare.20499630.v2](https://doi.org/10.6084/m9.figshare.20499630.v2) (2023).