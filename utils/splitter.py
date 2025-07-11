import os
import numpy as np
import anndata

class DataSplitter:
    def __init__(self, seed=42):
        """
        Initializes the DataSplitter with a random seed.

        Parameters:
        -----------
        seed : int, optional
            The random seed for reproducible results. Default is None.
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    """
    DataSplitter: A utility class for handling various data splitting tasks.

    This class provides static and class methods for splitting datasets 
    into train, test, validation sets, and performing other data partitioning tasks 
    like random selection, and even-odd based splits. The class also includes methods 
    for saving the partitioned data. 
    Note: This class was created using chatgpt4

    Methods:
    --------
    get_random_selection(lst: list) -> tuple:
        Shuffles and splits a list into two roughly equal halves.

    get_pairs_odds_selection(lst: list) -> tuple:
        Splits a list into even and odd numbers.

    get_adata(col_vals: list, col: str, adata: AnnData) -> AnnData:
        Retrieves subset of an AnnData object based on given column values.

    split_seen_unseen(col_vals: list, col: str, adata: AnnData, method: str) -> tuple:
        Splits data into seen and unseen sets either randomly or based on even-odd.

    get_train_val_test_indices(X: np.ndarray, train_ratio: float, val_ratio: float) -> tuple:
        Splits the data into train, validation, and test indices based on provided ratios.

    create_folder(folder_path: str):
        Creates a new folder at the specified path.

    save_adata(adata: AnnData, output_path: str):
        Saves an AnnData object's content as separate files.

    get_KFolds(adata: AnnData, col: str, out_folder: str):
        Divides an AnnData object into K Folds and saves each fold's train, validation, and test data.

    Note:
    -----
    The class is designed primarily for use with the AnnData object, but some methods can also 
    work with generic lists or numpy arrays.
    """

    @staticmethod
    def get_random_selection(lst):
        import random
        """
        Shuffles and splits a list into two roughly equal halves.

        Parameters:
        -----------
        lst: list
            The list to be split.

        Returns:
        --------
        tuple: Two lists, each being a subset of the original list.
        """
        random.shuffle(lst)
        half_length = len(lst) // 2
        return lst[:half_length], lst[half_length:]

    @staticmethod
    def get_pairs_odds_selection(lst):
        """
        Splits a list into even and odd numbers.

        Parameters:
        -----------
        lst: list
            The list containing numbers to be split.

        Returns:
        --------
        tuple: Two lists - first with even numbers and second with odd numbers from the input list.
        """
        pairs = [num for num in lst if num % 2 == 0]
        odds = [num for num in lst if num % 2 != 0]
        return pairs, odds

    @staticmethod
    def get_adata(col_vals: list, col: str, adata):
        """
        Retrieves subset of an AnnData object based on given column values.

        Parameters:
        -----------
        col_vals: list
            The values from the column based on which to subset the data.
        col: str
            The column name in the AnnData.obs to be checked.
        adata: AnnData
            The AnnData object to be subset.

        Returns:
        --------
        AnnData: Subset of the original AnnData object.
        """
        mask = adata.obs[col].isin(col_vals)
        return anndata.AnnData(X=adata.X[mask], obs=adata.obs[mask], var=adata.var)

    @classmethod
    def split_seen_unseen(self, col_vals: list, col: str, adata, method='random'):
        """
        Splits data into seen and unseen sets either randomly or based on even-odd.

        Parameters:
        -----------
        col_vals: list
            The values from the column based on which to subset the data.
        col: str
            The column name in the AnnData.obs to be checked.
        adata: AnnData
            The AnnData object to be subset.
        method: str, optional (default='random')
            The method to use for splitting. Choose between 'random' and 'pairs_odds'.

        Returns:
        --------
        tuple: Two AnnData objects - 'seen' and 'unseen'.
        """
        if method == 'random':
            seen_ids, unseen_ids = self.get_random_selection(col_vals)
        elif method == 'pairs_odds':
            seen_ids, unseen_ids = self.get_pairs_odds_selection(col_vals)
        else:
            raise ValueError("Invalid method provided. Choose 'random' or 'pairs_odds'.")

        seen_ = [col + "-{}".format(i) for i in seen_ids]
        unseen_ = [col + "-{}".format(i) for i in unseen_ids]

        seen_adata = self.get_adata(seen_, col, adata)
        unseen_adata = self.get_adata(unseen_, col, adata)
        return seen_adata, unseen_adata


    @staticmethod
    def get_train_val_test_indices(X, train_ratio=0.7, val_ratio=0.15):
        """
        Splits the data into train, validation, and test indices based on provided ratios.

        Parameters:
        -----------
        X: np.ndarray
            The data matrix to be split.
        train_ratio: float, optional (default=0.7)
            The ratio of data to be used for training.
        val_ratio: float, optional (default=0.15)
            The ratio of data to be used for validation.

        Returns:
        --------
        tuple: Indices for train, validation, and test sets.
        """
        from sklearn.model_selection import train_test_split
        # First, we split the data into training+validation set and test set
        train_val_X, test_X, train_val_idx, test_idx = train_test_split(X, 
                                                                        list(range(X.shape[0])), 
                                                                        test_size=1-train_ratio-val_ratio, 
                                                                        random_state=42)

        # Then, split the training+validation set into training and validation sets
        train_X, val_X, train_idx, val_idx = train_test_split(train_val_X, 
                                                              train_val_idx, 
                                                              test_size=val_ratio/(train_ratio+val_ratio), 
                                                              random_state=42)
        return train_idx, val_idx, test_idx

    @staticmethod
    def create_folder(folder_path):
        """
        Creates a new folder at the specified path.

        Parameters:
        -----------
        folder_path: str
            The path of the folder to be created.
        """
        if not os.path.exists(folder_path):
            print("creating folder:", folder_path)
            os.makedirs(folder_path)

    @staticmethod
    def save_adata(adata, output_path):
        """
        Saves an AnnData object's content as separate files.

        Parameters:
        -----------
        adata: AnnData
            The AnnData object to be saved.
        output_path: str
            The folder path where the AnnData object's content should be saved.
        """
        DataSplitter.create_folder(output_path)   
        np.save(output_path+'/exprMatrix.npy', adata.X)
        adata.var.to_csv(output_path+'/geneids.csv')
        adata.obs.to_csv(output_path+'/meta.csv')


    @staticmethod
    def get_KFolds(adata, out_folder, n_splits=10, stratified=False, stratify_cols=None):

        """
        Splits an AnnData object into training, validation, and test sets across K Folds for cross-validation. 
        It supports both stratified and non-stratified splits. The function ensures that each fold's test and 
        validation sets are unique, with the remaining data used for training.

        Returns:
        --------
        None
            The function saves the training, validation, and test datasets for each fold to the specified `out_folder`.
            The train/validation/test split for each fold is determined by the KFold (or StratifiedKFold) indices.

        Notes:
        ------
        - The function uses KFold (or StratifiedKFold if `stratified` is True) for the outer loop to split data into 
        test and train+validation sets. Then, an inner KFold is used to split train+validation into separate training 
        and validation sets.
        - The stratified split ensures balanced representation of classes specified in `stratify_cols` across all folds.
        - For each fold, the function saves the train, validation, and test sets as separate AnnData objects in the 
        specified `out_folder`.
        - train fraction is (n_splits-2)/n_splits
        - test and val fractions are 1/n_splits each
        """
        from sklearn.model_selection import KFold, StratifiedKFold

        # Assuming adata.X and adata.obs are already defined
        X = adata.X.copy()
        adata.obs["original_index"] = adata.obs.index.values
        y = adata.obs.copy()

        # Shuffle the indices manually for initial randomness

        indices_original = np.arange(X.shape[0])
        indices = indices_original.copy()
        np.random.shuffle(indices)

        # Subset X and y according to the shuffled indices
        X_shuffled = X[indices]
        y_shuffled = y.iloc[indices].copy()
        y_shuffled.reset_index(drop=True, inplace=True)



        # Prepare for stratified or regular KFold
        if stratified:
            if not stratify_cols:
                raise ValueError("stratify_cols must be specified when stratified is True.")
            # Creating a stratification group column
            y_shuffled['stratify_group'] = y_shuffled[stratify_cols].apply(lambda x: '_'.join(x.map(str)), axis=1)
            kfold_outer = StratifiedKFold(n_splits=n_splits, shuffle=False)
        else:
            kfold_outer = KFold(n_splits=n_splits, shuffle=False)

        # Initialize variable for special validation indices
        val_split_n_index = None

        # Outer loop for cross-validation
        fold_no = 1
        print("Note: indices and inner indices are independent")
        for train_val_index, test_index in kfold_outer.split(X_shuffled, y_shuffled['stratify_group'] if stratified else None):
            print(f"\nProcessing Fold {fold_no}")
            print("Train/Val indices:", train_val_index[:10])  # Print first 10 train/val indices
            print("Test indices:", test_index[:10])

            # Subset X and y for train/validation indices
            X_train_val = X_shuffled[train_val_index]
            y_train_val = y_shuffled.iloc[train_val_index].copy()
            y_train_val.reset_index(drop=True, inplace=True)
            # Assign values to X_test, y_test
            X_test = X_shuffled[test_index]
            y_test = y_shuffled.iloc[test_index].copy()
            y_test.reset_index(drop=True, inplace=True)

            # Saving the first fold's test set (to use it in the last fold for validation. Rotation strategy)
            if fold_no == 1:
                val_split_n_index = test_index

            # Adjusting train/val split for the last fold
            if fold_no == n_splits and val_split_n_index is not None:
                train_index = np.setdiff1d(train_val_index, val_split_n_index)
                val_index = val_split_n_index
                # Split data for train, validation, and test sets
                X_train, X_val = X_shuffled[train_index], X_shuffled[val_index]

                y_train, y_val = y_shuffled.iloc[train_index], y_shuffled.iloc[val_index]
                y_train.reset_index(drop=True,inplace=True)
                y_val.reset_index(drop=True,inplace=True)
                print("Special case for the last fold:")
                print("Train indices:", train_index[:10])  # Print first 10 train indices
                print("Val indices:", val_index[:10])      # Print first 10 validation indices
            else:
                # Inner loop for nested cross-validation
                inner_fold_no = 1
                kfold_inner = StratifiedKFold(n_splits=n_splits-1, shuffle=False) if stratified else KFold(n_splits=n_splits-1, shuffle=False)
                for inner_train_index, inner_val_index in kfold_inner.split(X_train_val, y_train_val['stratify_group'] if stratified else None):
                    if fold_no == inner_fold_no:
                        # Split data for train and validation sets
                        X_train, X_val = X_train_val[inner_train_index], X_train_val[inner_val_index]
                        y_train, y_val = y_train_val.iloc[inner_train_index], y_train_val.iloc[inner_val_index]
                        y_train.reset_index(drop=True,inplace=True)
                        y_val.reset_index(drop=True,inplace=True)
                        print(f"Inner Fold {inner_fold_no}")
                        print("Inner Train indices:", inner_train_index[:10])  # Print first 10 inner train indices
                        print("Inner Val indices:", inner_val_index[:10])      # Print first 10 inner val indices
                        break
                    inner_fold_no += 1


            # Saving the split data
            splits_folder = os.path.join(out_folder, f"split_{fold_no}")
            os.makedirs(splits_folder, exist_ok=True)

            adata_train = anndata.AnnData(X=X_train, obs=y_train, var=adata.var)
            adata_val = anndata.AnnData(X=X_val, obs=y_val, var=adata.var)
            adata_test = anndata.AnnData(X=X_test, obs=y_test, var=adata.var)

            DataSplitter.save_adata(adata_train, os.path.join(splits_folder, "train"))
            DataSplitter.save_adata(adata_test, os.path.join(splits_folder, "test"))
            DataSplitter.save_adata(adata_val, os.path.join(splits_folder, "val"))

            fold_no += 1

    @staticmethod
    def get_KFolds_likeSon(adata, out_folder, n_splits=10, stratified=False, stratify_cols=None):
        from sklearn.model_selection import KFold, StratifiedKFold
        # 1. Reshuffle data
        # Create a copy from original X, y. The copies will be reshuffled
        X = adata.X.copy()
        adata.obs["original_index"] = adata.obs.index.values
        y = adata.obs.copy()

        # Shuffle the indices manually for initial randomness
        indices_original = np.arange(X.shape[0])
        indices = indices_original.copy()
        np.random.shuffle(indices)

        # Subset X and y according to the shuffled indices
        X_shuffled = X[indices]
        y_shuffled = y.iloc[indices].copy()
        y_shuffled.reset_index(drop=True, inplace=True)

        # 2. Define outer KFold
        if stratified:
            if not stratify_cols:
                raise ValueError("stratify_cols must be specified when stratified is True.")
            y_shuffled['stratify_group'] = y_shuffled[stratify_cols].apply(lambda x: '_'.join(x.map(str)), axis=1)
            kfold_outer = StratifiedKFold(n_splits=n_splits, shuffle=True)  # Shuffle set to True even though we are shuffling before
        else:
            kfold_outer = KFold(n_splits=n_splits, shuffle=True)  # Shuffle set to True even though we are shuffling before

        # 3. For loop to Split data in each fold
        fold_no = 1
        for train_val_index, test_index in kfold_outer.split(X_shuffled, y_shuffled['stratify_group'] if stratified else None):
        # The rest of your method remains the same, just make sure to use X_shuffled and y_shuffled

            print(f"Processing Fold {fold_no}")

            # Subset X and y for train_val indices
            X_train_val, y_train_val = X_shuffled[train_val_index], y_shuffled.iloc[train_val_index].copy()
            y_train_val.reset_index(drop=True, inplace=True)

            # 3.1. Assign values to X_test, y_test
            X_test, y_test = X_shuffled[test_index],y_shuffled.iloc[test_index].copy()
            y_test.reset_index(drop=True, inplace=True)

            # 3.2.1 Define inner KFold for train/validation split
            if stratified:
                #I do not want leakage in train/val across folds. That is why, we will not shuffle again. However we did shuffle all data at the beginning of this method.
                kfold_inner = StratifiedKFold(n_splits=n_splits-1, shuffle=False)
                train_index, val_index = next(kfold_inner.split(X_train_val, y_train_val['stratify_group']))
            else:
                kfold_inner = KFold(n_splits=n_splits-1, shuffle=False)
                train_index, val_index = next(kfold_inner.split(X_train_val))

            # 3.2.2. Extract data for train, validation
            X_train, X_val = X_train_val[train_index], X_train_val[val_index]
            y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]
            
            # 4. Saving the split data
            splits_folder = os.path.join(out_folder, f"split_{fold_no}")
            os.makedirs(splits_folder, exist_ok=True)

            adata_train = anndata.AnnData(X=X_train, obs=y_train, var=adata.var)
            adata_val = anndata.AnnData(X=X_val, obs=y_val, var=adata.var)
            adata_test = anndata.AnnData(X=X_test, obs=y_test, var=adata.var)

            DataSplitter.save_adata(adata_train, os.path.join(splits_folder, "train"))
            DataSplitter.save_adata(adata_test, os.path.join(splits_folder, "test"))
            DataSplitter.save_adata(adata_val, os.path.join(splits_folder, "val"))

            fold_no += 1


    @staticmethod
    def generate_fold_indices(X, n_splits=5):
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=True)  # Uses the global random state set by np.random.seed
        fold_indices = [test_index for _, test_index in kf.split(X)]
        return fold_indices

    @staticmethod    
    def generate_stratified_fold_indices(X, y, n_splits=5):

        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)  # Uses the global random state set by np.random.seed
        fold_indices = [test_index for _, test_index in skf.split(X, y)]
        return fold_indices

    @staticmethod
    def rotate_folds_for_cross_validation(fold_indices):
        """
        Rotate through fold indices to assign training, testing, and validation sets.
        """
        n_splits = len(fold_indices)
        for i in range(n_splits):
            test_fold = i
            val_fold = (i + 1) % n_splits  # Wrap-around at the end
            train_folds = [fold_indices[j] for j in range(n_splits) if j != test_fold and j != val_fold]
            train_idx = np.concatenate(train_folds)
            test_idx = fold_indices[test_fold]
            val_idx = fold_indices[val_fold]
            yield train_idx, test_idx, val_idx
    @staticmethod
    def Get_Kfolds_likeAlbert(adata, out_folder, n_splits=10, stratified=False, stratify_cols=None):
        """
        Splits an AnnData object into training, validation, and test sets across K Folds for cross-validation.
        """
        # Shuffle the indices manually for initial randomness
        indices_original = np.arange(adata.shape[0])
        np.random.shuffle(indices_original)
        adata.obs["original_index"] = adata.obs.index.values

        # Subset adata according to the shuffled indices
        adata_shuffled = adata[indices_original].copy()
        
        X = adata_shuffled.X
        y = None
        if stratified:
            if stratify_cols is not None:
                y = adata_shuffled.obs[stratify_cols].astype(str).apply('_'.join, axis=1)
                # Example of combining 'donor' and 'celltype' into a single label

            else:
                raise ValueError("stratify_cols must be specified when stratified is True.")
        
        # Choose the fold generation method based on the stratified parameter
        if stratified and y is not None:
            fold_indices = DataSplitter.generate_stratified_fold_indices(X, y, n_splits=n_splits)
        else:
            fold_indices = DataSplitter.generate_fold_indices(X, n_splits=n_splits)
        
        # Rotate through the folds and create the splits
        for fold_no, (train_idx, test_idx, val_idx) in enumerate(DataSplitter.rotate_folds_for_cross_validation(fold_indices), start=1):
            print(f"\nProcessing Fold {fold_no}")
            X_train, y_train = X[train_idx], adata_shuffled.obs.iloc[train_idx]
            X_test, y_test = X[test_idx], adata_shuffled.obs.iloc[test_idx] 
            X_val, y_val = X[val_idx], adata_shuffled.obs.iloc[val_idx] 
            
            # Save the split data
            splits_folder = os.path.join(out_folder, f"split_{fold_no}")
            os.makedirs(splits_folder, exist_ok=True)
            
            adata_train = anndata.AnnData(X=X_train, obs=y_train)
            adata_test = anndata.AnnData(X=X_test, obs=y_test)
            adata_val = anndata.AnnData(X=X_val, obs=y_val)

            DataSplitter.save_adata(adata_train, os.path.join(splits_folder, "train"))
            DataSplitter.save_adata(adata_test, os.path.join(splits_folder, "test"))
            DataSplitter.save_adata(adata_val, os.path.join(splits_folder, "val"))
            

    @staticmethod
    def check_stratification(adata: anndata.AnnData, adata_train: anndata.AnnData, 
                            adata_val: anndata.AnnData, adata_test: anndata.AnnData, 
                            stratify_cols: list):
        """
        Check if the training, validation, and test data are stratified based on the given columns.

        Parameters:
        -----------
        adata : anndata.AnnData
            The original AnnData object.
        adata_train : anndata.AnnData
            The AnnData object for the training set.
        adata_val : anndata.AnnData
            The AnnData object for the validation set.
        adata_test : anndata.AnnData
            The AnnData object for the test set.
        stratify_cols : list
            The columns in the 'obs' DataFrame to check stratification for.

        Returns:
        --------
        comparison : pd.DataFrame
            A DataFrame showing the distribution of combined values in the stratify_cols for the original, 
            training, validation, and test datasets.
        """
        import pandas as pd
        # Create combined stratify column for original and split data
        adata.obs['combined_stratify_col'] = adata.obs[stratify_cols].apply(lambda x: '_'.join(x.map(str)), axis=1)
        adata_train.obs['combined_stratify_col'] = adata_train.obs[stratify_cols].apply(lambda x: '_'.join(x.map(str)), axis=1)
        adata_val.obs['combined_stratify_col'] = adata_val.obs[stratify_cols].apply(lambda x: '_'.join(x.map(str)), axis=1)
        adata_test.obs['combined_stratify_col'] = adata_test.obs[stratify_cols].apply(lambda x: '_'.join(x.map(str)), axis=1)

        # Compute distribution in original data
        original_dist = adata.obs['combined_stratify_col'].value_counts(normalize=True)

        # Compute distribution in train, val, and test data
        train_dist = adata_train.obs['combined_stratify_col'].value_counts(normalize=True)
        val_dist = adata_val.obs['combined_stratify_col'].value_counts(normalize=True)
        test_dist = adata_test.obs['combined_stratify_col'].value_counts(normalize=True)

        # Compare distributions
        comparison = pd.DataFrame({
            'Original': original_dist,
            'Train': train_dist,
            'Validation': val_dist,
            'Test': test_dist
        }).fillna(0)  # Fill NA values with 0, in case a category isn't present in a subset

        return comparison




