
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from anndata import AnnData
from matplotlib.lines import Line2D


import time
#import scanpy as sc



def calculate_merge_scores(latent_list, adata, labels,sample_size=None):
    """
    Calculates clustering scores for multiple latent representations using the provided labels,
    and then aggregates them into a single DataFrame.
    """
    # Initialize a list to hold the DataFrame rows before concatenation
    scores_list = []

    # Iterate over each latent representation and calculate clustering scores
    for latent in latent_list:
        # Assuming get_clustering_scores_optimized returns a DataFrame with the scores
        scores = get_clustering_scores_optimized(adata, latent, labels,sample_size)
        print("\nscores",scores)
        
        # Assuming restructure_dataframe restructures the scores DataFrame as needed for aggregation
        scores_row = restructure_dataframe(scores, labels)
        print("\nscores row",scores_row)
        
        # Append the restructured row to the list
        scores_list.append(scores_row)

    # Concatenate all DataFrame rows at once outside the loop
    combined_df = pd.concat(scores_list, ignore_index=True)

    # Assign the latent representation names as the DataFrame index
    combined_df.index = latent_list

    return combined_df


def restructure_dataframe(df, labels):
    """
    Restructures a given dataframe based on specific clustering scores and labels, creating a multi-level column format.
    It is useful to restructure the output of calculate_merge_scores.

    Parameters:
    - df (pd.DataFrame): Input dataframe that contains clustering scores.
    - labels (list): List of label columns (e.g., 'donor', 'celltype') that will be used to reorder and index the dataframe.

    Returns:
    - new_df (pd.DataFrame): Restructured dataframe with multi-level columns based on the unique index names and original columns.
                             The dataframe contains clustering scores for the specified labels.
    
    Notes:
    The function specifically looks for the clustering scores "ch", "1/db", and "silhouette" in the input dataframe.
    """

    # reorder
    df = df.loc[["ch","1/db","silhouette"],labels].T
    # Get the unique index names (donor, celltype, etc.)
    index_names = df.index.unique().tolist()

    # Extracting the column names
    cols = df.columns.tolist()

    # Creating multi-level column names based on index_names and cols
    new_columns = pd.MultiIndex.from_product([index_names, cols])

    # Flatten the data for the new structure
    values = df.values.flatten()

    # Creating the new DataFrame
    new_df = pd.DataFrame([values], columns=new_columns)
    return new_df




def reshape_scores(df_scores: pd.DataFrame,
                   labels: list[str],
                   fold: int,
                   dataset_type: str) -> pd.DataFrame:
    """
    Parameters
    ----------
    df_scores
        A DataFrame whose columns already form a MultiIndex
        (level 0 = label, level 1 = metric) ? e.g. the output of
        `calculate_merge_scores`.
    labels
        The subset and order of label columns you want to keep
        (e.g. ['batch', 'celltype']).
    fold
        Fold number (integer) to record in the row.
    dataset_type
        String describing the split or latent (will become a column).

    Returns
    -------
    pd.DataFrame
        One-row frame with:
          * multi-index columns (label, metric) **unchanged**, and
          * extra single-level columns ?fold? and ?dataset_type?.
    """

    # Metrics we care about, in the order we want them
    metrics = ["ch", "1/db", "silhouette"]

    # Pick the desired columns and **preserve** the MultiIndex structure
    wanted_cols = pd.MultiIndex.from_product([labels, metrics])
    sub = df_scores[wanted_cols].copy()

    # In most cases you only need one row (take the first);
    # change `.iloc[[0]]` to something else if you want to aggregate.
    row_df = sub.iloc[[0]].reset_index(drop=True)

    # Append bookkeeping columns without touching the MultiIndex
    row_df["fold"] = fold
    row_df["dataset_type"] = dataset_type

    # Put the simple columns at the far right (optional)
    row_df = row_df.reindex(columns=list(wanted_cols) + ["fold", "dataset_type"])

    return row_df




def plot_rep(
    adata: AnnData,
    shape_col: str = "celltype",
    color_col: str = "celltype",
    use_rep: str = "X_pca",
    markers: tuple = ("o", "v", "^", "<", "*"),
    clustering_scores=None,
    save_fig: bool = True,
    outpath: str = "",
    showplot: bool = False,
    palette_choice: str = "tab20",
    file_name: str = "latent",
    figsize: tuple = (7, 7)):
    """Plot 2D embedding with separate colour and shape legends."""
    print("Plotting latent representation:", use_rep)

    plt.ioff()
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # --------------------------------------------
    # Determine unique categories
    # --------------------------------------------
    unique_shapes = np.unique(adata.obs[shape_col])
    unique_colors = np.unique(adata.obs[color_col])

    # --------------------------------------------
    # Build colour palette
    # --------------------------------------------
    if isinstance(palette_choice, list):
        color_palette = palette_choice
    elif palette_choice == "hsv":
        color_palette = sns.color_palette("hsv", len(unique_colors))
    elif palette_choice == "tab20":
        color_palette = [plt.cm.tab20(i) for i in np.linspace(0, 1, len(unique_colors))]
    elif palette_choice == "Set2":
        color_palette = sns.color_palette("Set2", len(unique_colors))
    else:
        raise ValueError("palette_choice must be 'hsv', 'tab20', 'Set2', or a list of colours")

    color_map = {c: color_palette[i] for i, c in enumerate(unique_colors)}
    shape_map = {s: markers[i % len(markers)] for i, s in enumerate(unique_shapes)}

    # --------------------------------------------
    # Coordinates
    # --------------------------------------------
    coords = adata.X if use_rep == "X" else adata.obsm[use_rep]
    x, y = coords[:, 0], coords[:, 1]

    for s in unique_shapes:
        for c in unique_colors:
            mask = (adata.obs[shape_col] == s) & (adata.obs[color_col] == c)
            ax.scatter(
                x[mask],
                y[mask],
                color=color_map[c],
                marker=shape_map[s],
                alpha=0.7,
                s=1,
            )

    # --------------------------------------------
    # Legends
    # --------------------------------------------
    colour_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color_map[c], markeredgecolor="none", markersize=10)
        for c in unique_colors
    ]
    legend_colour = ax.legend(
        handles=colour_handles,
        labels=list(unique_colors),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title=color_col,
    )
    ax.figure.add_artist(legend_colour)

    shape_handles = [
        Line2D([0], [0], marker=shape_map[s], linestyle="", color="black", markerfacecolor="none", markersize=10, markeredgewidth=1.5)
        for s in unique_shapes
    ]
    legend_shape = ax.legend(
        handles=shape_handles,
        labels=list(unique_shapes),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.65),
        ncol=3,
        title=shape_col,
    )

    extra_artists = (legend_colour, legend_shape)

    # --------------------------------------------
    # Axis labels
    # --------------------------------------------
    if "pca" in use_rep and "variance_ratio" in adata.uns.get("pca", {}):
        vr = adata.uns["pca"]["variance_ratio"]
        ax.set_xlabel(f"PC 1 ({vr[0]*100:.2f} %)")
        ax.set_ylabel(f"PC 2 ({vr[1]*100:.2f} %)")
    else:
        ax.set_xlabel(f"{use_rep} 1")
        ax.set_ylabel(f"{use_rep} 2")

    # --------------------------------------------
    # Save / show
    # --------------------------------------------
    if save_fig and outpath:
        os.makedirs(outpath, exist_ok=True)
        fig.savefig(
            os.path.join(outpath, f"{use_rep}_{file_name}.png"),
            bbox_extra_artists=extra_artists,
            bbox_inches="tight",
        )
    if showplot:
        plt.show()
    else:
        plt.close(fig)


def get_clustering_scores_optimized(adata, use_rep, labels, sample_size=None):
    from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
    """
    Optimized version of getting clustering scores from an AnnData object with optional subsampling for all scores.

    Args:
    - adata: An AnnData object.
    - use_rep: Representation to use. Any key for .obsm is valid. If 'X', then adata.X is used.
    - labels: Labels for clustering. Any key for .obs is valid.
    - sample_size: Optional integer. If specified, subsamples this number of instances for score calculation.

    Returns:
    - A pandas DataFrame with Davies-Bouldin (DB), inverse DB, Calinski-Harabasz (CH), and Silhouette scores for each label.
    """
    
    start_time = time.time()
    print("Computing scores..")
    if sample_size is not None:
        print("sample_size:",sample_size)
    else:
        print("all cells used")
        
    print("\nDB: lower values --> better clustering")
    print("1/DB, CH, Silhouette: higher values --> better clustering")

    # Determine the data representation
    data_rep = adata.X if use_rep == "X" else adata.obsm[use_rep]

    dict_scores = {}
    time_per_score = {}

    # Subsample the data if a sample size is specified
    if sample_size and sample_size < data_rep.shape[0]:
        indices = np.random.choice(data_rep.shape[0], sample_size, replace=False)
        subsampled_data_rep = data_rep[indices]
    else:
        subsampled_data_rep = data_rep

    # Compute scores for each label and store in dict
    for label in labels:
        labels_array = adata.obs[label].to_numpy()
        if sample_size and sample_size < data_rep.shape[0]:
            subsampled_labels = labels_array[indices]
        else:
            subsampled_labels = labels_array

        score_start_time = time.time()
        db_score = davies_bouldin_score(subsampled_data_rep, subsampled_labels)
        db_time = time.time() - score_start_time

        score_start_time = time.time()
        ch_score = calinski_harabasz_score(subsampled_data_rep, subsampled_labels)
        ch_time = time.time() - score_start_time

        score_start_time = time.time()
        silhouette = silhouette_score(subsampled_data_rep, subsampled_labels)
        silhouette_time = time.time() - score_start_time

        dict_scores[label] = [db_score, 1/db_score, ch_score, silhouette]
        time_per_score[label] = [db_time, ch_time, silhouette_time]

    # Create DataFrame from dict
    df_scores = pd.DataFrame(dict_scores, index=['db', '1/db', 'ch', 'silhouette'])
    total_time = time.time() - start_time

    print(f"Total computation time: {total_time} seconds")
    for label, times in time_per_score.items():
        print(f"Time per score for {label} - DB: {times[0]:.4f}, CH: {times[1]:.4f}, Silhouette: {times[2]:.4f} seconds")

    return df_scores

