# Define general data paths to the all the simulation outputs

data_path = "/archive/bioinformatics/DLLab/AixaAndrade/data/Genomic_data/ASD/reverse_norm"

# Path data to split (path to the location of the data)

data2split_foldername = "log_transformed_2916hvggenes"

import os
# folder in which the splits will be saved. They are saved inside of data2split_foldername. Please create your own paths to store data to avoid rewriting
folder_splits = os.path.join(data2split_foldername,"splits")
#folder_splits_unseen_pairs_odds = data2split_foldername  +"/unseen_splits_pairs_odds"
