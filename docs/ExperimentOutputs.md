
# Experiment Outputs Structure

After running your model, the results are stored in the `outputs/<datasetname>_outputs` directory. This directory is organized by scenario, run, and split, and includes latent space embeddings, figures, saved models, and performance metrics. The MEC model has a distinct output structure, focusing on classification metrics (primarily from the random forest) rather than latent embeddings and reconstruction metrics.


**Note:**  
- The `latent_space`, `figures`, and `saved_models` directories are generated only by running the AE, AEC, scMEDAL-FE, scMEDAL-FEC, or scMEDAL-RE models.  
- The UMAP and genomap visualizations and the MEC model require these outputs, as they cannot run without them.  
---

## Latent Space Outputs

### For AE, AEC, scMEDAL-FE, scMEDAL-FEC, and scMEDAL-RE  Models

```markdown


outputs/
|-- <datasetname>_outputs/                                         # Directory containing all output results for a given dataset
    |-- latent_space/                                              # Contains latent representations and associated metrics
        |-- <scenario_id>/                                         # Represents a particular experimental scenario or configuration
            |-- <modelname>/                                       # Specifies the model used (e.g., AE, scMEDAL-FE, scMEDAL-RE, etc.)
                |-- <run_name>/                                    # A unique identifier for a particular run of the model

                    # Global (aggregated) output files across all folds:
                    |-- all_scores_<data_type>_samplesize-<sample_size>.csv  # Consolidated ASW, DB, CH scores from all splits
                    |-- history_allfolds.csv                                # Aggregated training history metrics from all splits
                    |-- mean_history_allfolds.csv                           # Average training metrics computed across all folds
                    |-- mean_scores_<data_type>_samplesize-<sample_size>.csv # Average ASW, DB, CH scores across all folds and data types

                    # Outputs for each data split (e.g., splits_1/):
                    |-- splits_i/                                        # Directory containing results for the i-th split
                        |-- <encoder_latent_name>_<data_type>.npy        # Latent embeddings (used for downstream analyses like UMAP)
                        |-- recon_<data_type>.npy                        # Reconstructed data corresponding to the i-th split
                        |-- scores_<data_type>_samplesize-<sample_size>.csv # ASW, DB, CH scores specific to this split and data type
                        |-- history.csv                                  # Detailed training history (loss per epoch) for this split

```


**Additional Files for scMEDAL-RE (`get_cf_batch=True`):**  
If `get_cf_batch=True` in `model_config.py`, batch-specific reconstructions are generated for each batch `i`:

- `recon_batch_train_<batch_i>.npy`
- `recon_batch_val_<batch_i>.npy`
- `recon_batch_test_<batch_i>.npy`

These files allow analysis of how each cell would be reconstructed as if it originated from a specific batch, aiding in batch effect studies.

---

### For MEC Model

The MEC model produces a different set of outputs focused on classification metrics rather than latent embeddings and reconstruction metrics:

```markdown
outputs/
|-- <datasetname>_outputs/
    |-- latent_space/
        |-- <scenario_id>/
            |-- <modelname>/
                |-- <run_name>/   
                    # Aggregated metrics across folds:
                    |-- metrics_allfolds.csv          # RFAccuracy, RFBalancedAccuracy, ChanceAccuracy
                    |-- metrics_allfolds_95CI.csv     # Above metrics with 95% confidence intervals
                    # Per-split outputs (e.g., splits_i/):
                    |-- splits_i/
                        |-- metrics.csv               # Metrics for this single fold
                        |-- y_pred_<data_type>_dummy.csv # Dummy classifier predictions (for chance accuracy)
                        |-- y_pred_test_rf.csv         # Random forest classifier predictions (primary method used)


```

---

## Figure Outputs

### For AE, AEC, scMEDAL-FE, scMEDAL-FEC, and scMEDAL-RE Models

```markdown
outputs/
|-- <datasetname>_outputs/
    |-- figures/
        |-- <scenario_id>/
            |-- <modelname>/
                |-- <run_name>/
                    |-- avg_loss.png  # Average loss across all folds
                    |-- splits_i/
                        # Latent space plot colored by bio_col (cell type)
                        |-- <encoder_latent_name>_<data_type>_<bio_col>-<bio_col>_<modelname>_latent_<data_type>.png 
                        # Latent space plot colored by batch
                        |-- <encoder_latent_name>_<data_type>_<donor_col>-<donor_col>_<modelname>_latent_<data_type>.png 
                        |-- X_pca_<data_type>_<bio_col>-<bio_col>_<data_type>.png
                        |-- loss.png    # Training loss plot for this fold

```
- data_type in [train, val, test]
- <bio_col>, <donor_col> represent biological or donor annotations
### For MEC Model


```markdown
outputs/
|-- <datasetname>_outputs/
    |-- latent_space/
        |-- <scenario_id>/
            |-- <modelname>/
                |-- <run_name>/
                    |-- splits_i/
                        |-- loss.png  # DFFN classifier loss plot (typically not used)
```


**Note:** These figures are mainly for initial quality checks (e.g., monitoring training progress) and may need refinement for publication or presentation.

---

## Saved Models

For all model types (AE, AEC, scMEDAL-FE, scMEDAL-FEC, scMEDAL-RE), the following structure is used. 

```markdown
outputs/
|-- <datasetname>_outputs/
    |-- saved_models/
        |-- <scenario_id>/
            |-- <modelname>/
                |-- <run_name>/
                    |-- model_config.py    # The model configuration used during training

                    # Checkpoints per split (e.g., splits_i/):
                    |-- splits_i/
                        |-- checkpoint
                        |-- cp-0000.ckpt.data-00000-of-00001
                        |-- cp-0000.ckpt.index
                        |-- cp-0001.ckpt.data-00000-of-00001
                        |-- cp-0001.ckpt.index
                        |-- cp-0002.ckpt.data-00000-of-00001
                        |-- cp-0002.ckpt.index
                        |-- model_params.yaml
```
**Note:**
- Each `splits_i/` directory contains model checkpoints saved at regular intervals.
- Use these checkpoints to resume training or load the model for inference.

---

## Compare Models Outputs

To be able to compare model outputs, you first update the output folders, see [Outputs and Analysis folders](../README.md#outputs-and-analysis-folders) run the following lines of code.
```python
import analysis.analysis as aa

analysis_name = "AML_demo"

aml = aa.AMLAnalysis(model_folder_dict, analysis_name)

res= aml.clustering_scores(model_folder_dict)
```
See  [`demo/demo_aml.ipynb`](../demo/demo_aml.ipynb) for a clear example.

### Clustering Scores

After training your models, you may want a more convenient format for comparing clustering metrics (ASW, CH, DB) across folds.



**File Outputs:**


```markdown
outputs/
|-- <datasetname>_outputs/
    |-- compare_models/
        |-- <scenario_id>/
            |-- <run_name>/
                |-- <dataset_type>_allscores.csv                  # All scores (ASW, CH, DB) from all folds for <dataset_type>
                |-- <dataset_type>_scores_<sample_size>_95CI.csv  # Average ASW, CH, DB with 95% CI across folds
                |-- <dataset_type>_<sample_size>.csv              # Mean, SEM, STD of scores across folds
                |-- <dataset_type>_scores_<sample_size>_min_silhouette_batch.csv  # Fold with the lowest ASW
                |-- <dataset_type>_scores_<sample_size>_max_silhouette_batch.csv  # Fold with the highest ASW
```

**Explanation of Variables:**
- `<dataset_type>`: Specifies which dataset split is being evaluated (`train`, `val`, or `test`). Default is `test`.
- `<sample_size>`: Number of cells sampled for the analysis. Larger sample sizes give more robust metrics but require more computation.

These files help you quickly identify how well your models are clustering cells under various conditions and can guide model selection or hyperparameter tuning.

---

### Genomaps

Genomaps visualize gene-level variations across cells and batches, providing insights into how genes map onto a pixelated 2D space. This is useful for understanding batch effects and biological variability at a more granular level.

**File Outputs:**


```markdown
outputs/
|-- <datasetname>_outputs/
    |-- compare_models/
        |-- <scenario_id>/
            |-- <run_name>/
                |-- CMmultibatch_<n_cells_per_m_batches>_<Type>_<Split>_<celltype_name>_with_<n_inputs_fe>fe_input/
                    |-- exprMatrix.npy                # Expression matrix used to generate the genomap
                    |-- geneids.csv                   # List of gene IDs corresponding to the rows in exprMatrix
                    |-- meta.csv                      # Metadata for the cells (e.g., batch, cell type)

                |-- <n_cells_per_m_batches>_<Type>_<Split>_<celltype_name>_with_<n_inputs_fe>fe_input/
                    |-- <n_cells_per_m_batches>_<Type>_<Split>_<celltype_name>_with_<n_inputs_fe>fe_input.png  # Genomap visualization for the first 50 cells
                    |-- gene_coordinates_<n_cells_per_m_batches>_<Type>_<Split>_<celltype_name>_with_<n_inputs_fe>fe_input.csv # Coordinates mapping genes to pixels
                    |-- genomap_<n_cells_per_m_batches>_<Type>_<Split>_<celltype_name>_with_<n_inputs_fe>fe_input.npy  # Genomap data (2D representation)
                    |-- T_input_<n_cells_per_m_batches>_<Type>_<Split>_<celltype_name>_with_<n_inputs_fe>fe_input.npy  # Transformation matrix used in genomap generation
                    |-- genomap_10topvariablegenesacrossbatches_b'<cell_id>'_std.csv  # The top 10 variable genes for a particular cell across all batches
                    |-- genomap_10topvariablegenesacrossbatches_b'<cell_id>'_std_<celltype_name>.png  # Genomap highlighting top 10 variable genes
                    |-- genomap_10topvariablegenesacrossbatches_b'<cell_id>'_few_batches_std_<celltype_name>.png  # Genomap focusing on a subset of batches
                    |-- genomap_10topvariablegenesacrossbatches_b'<cell_id>'_few_batches_std_<celltype_name>_labels.png  # Genomap with labeled top 10 variable genes
```

**Explanation of Variables:**
- `<n_cells_per_m_batches>`: Combination of `<n_cells_per_batch>` and `<n_batches>` describing how cells are sampled.
- `<Type>`: The dataset split type (e.g., `train`).
- `<Split>`: The fold number (e.g., `2`).
- `<celltype_name>`: Concatenation of cell types (e.g., `Ventricular_Cardiomyocyte_Endothelial_Fibroblast_Pericytes`).
- `<n_inputs_fe>`: Number of additional features used (e.g., `2`).

These genomap files allow you to investigate how genes vary across conditions and can be used to pinpoint batch effects or identify biologically relevant gene patterns.

---

### UMAPs

UMAP (Uniform Manifold Approximation and Projection) provides a non-linear dimensionality reduction that preserves local and global data structure. It is used to visualize cell embeddings and identify potential clusters or batch effects.

**File Outputs:**

```markdown
outputs/
|-- <datasetname>_outputs/
    |-- compare_models/
        |-- <scenario_id>/
            |-- <run_name>/
                |-- umap_<n_batches_sample>batches/
                    |-- umap_<model>_<data_type>_split<Split>_<sample_size>cells_<n_batches_sample>batches.csv  # UMAP coordinates
                    |-- umap_<model>_<data_type>_split<Split>_<sample_size>cells_<n_batches_sample>batches.png  # UMAP plot colored by cell type
                    |-- umap_<model>_<data_type>_split<Split>_<sample_size>cells_<n_batches_sample>batches_batch.png  # UMAP plot colored by batch
```

**Explanation of Variables:**
- `<model>`: The model that generated the embeddings (e.g., `AE_latent`, `AEC_latent`, `scMEDAL-FE_latent`).
- `<data_type>`: The dataset split (`train`, `val`, `test`).
- `<Split>`: The fold number.
- `<sample_size>`: Number of cells sampled for the UMAP.
- `<n_batches_sample>`: Number of batches sampled (useful for large datasets with many batches).

**CSV Files:**
- Contain UMAP coordinates (UMAP1, UMAP2) for each cell, which can be further analyzed or plotted.

**PNG Files:**
- Visual representations of the UMAP embeddings, typically colored by cell type or batch. These plots help assess how well embeddings separate cell types or mix batches.

---

By using the `compare_clustering_scores.py` script and examining the genomaps and UMAP outputs, you can gain comprehensive insights into how well your models cluster cells, handle batch effects, and reveal underlying biological structure. These comparison outputs are essential for model selection, performance evaluation, and deeper exploratory analysis of your results.