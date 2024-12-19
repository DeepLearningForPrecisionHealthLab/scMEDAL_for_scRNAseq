import time
import numpy as np
import pandas as pd
import os
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
#from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.callbacks import Callback

def get_clustering_scores(X, labels, sample_size=None):
    """ Computes and returns clustering scores more efficiently. """

    if sample_size is not None: #and sample_size < X.shape[0]:
        indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X = X[indices]
        labels = labels[indices]

    db_score = davies_bouldin_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    
    return db_score, 1 / db_score, ch_score

def compute_latents_and_scores(model, X, metadata, labels_list, output_dir, epoch, sample_size=None,batch_size=32,model_type="ae_da"):
    """
    Computes latents and clustering scores, managing file I/O and data processing efficiently.
    
    Parameters:
    - model: The machine learning model to predict from.
    - X: Input features for prediction.
    - metadata: DataFrame containing labels for clustering scores.
    - labels_list: List of column names from metadata to compute scores for.
    - output_dir: Directory path to save outputs.
    - epoch: Current epoch number (zero-based index).
    - sample_size: Optional; number of samples to use for score computation.
    
    Returns:
    - Dictionary of scores data.
    """



    def get_input_data(input_data, model_type):
        """Helper function to determine input based on model type and data structure."""
        # if tuple x,z = input_data
        if isinstance(input_data, tuple):
            # for ae_re we use x,z as encoder inputs, else we use x only 
            return input_data if model_type == "ae_re" else input_data[0]
        return input_data

    X = get_input_data(X, model_type)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Get latent space
    if model_type == "ae_re":
        latent = model.re_encoder.predict(X, batch_size=batch_size) 
    else:
        # print("real use_layer_activations:",  model.encoder.return_layer_activations)

        # Check if the encoder object has the 'return_layer_activations' attribute and if it is True
        use_layer_activations = hasattr(model.encoder, 'return_layer_activations') and model.encoder.return_layer_activations

        # print("use_layer_activations:", use_layer_activations)

        outputs = model.encoder.predict(X, batch_size=batch_size)

        latent = outputs[-1] if use_layer_activations else outputs

        # print(latent,"latent")

    epoch_str = f'epoch{epoch+1:03d}'
    pd.DataFrame(latent).to_pickle(os.path.join(output_dir, f'{epoch_str}_latents.pkl'))

    scores_data = {}
    for label_col in labels_list:
        labels = metadata[label_col]
        #get clustering scores on latent space
        scores = get_clustering_scores(latent, labels, sample_size)
        scores_data.update({f'{label_col}_DB': scores[0], f'{label_col}_1/DB': scores[1], f'{label_col}_CH': scores[2]})
        score_path = os.path.join(output_dir, f'clustering_scores_{label_col}.csv')
        with open(score_path, 'a') as file:
            if epoch == 0:  # Check if header needs to be written
                file.write('epoch,DB,1/DB,CH\n')
            file.write(f'{epoch+1},{scores[0]},{scores[1]},{scores[2]}\n')

    return scores_data


class ComputeLatentsCallback(Callback):
    def __init__(self, model, X, metadata, labels_list, output_dir, sample_size=None, batch_size=32,model_type="ae_da"):
        self.model = model
        self.X = X
        self.metadata = metadata
        self.labels_list = labels_list
        self.output_dir = output_dir
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.model_type = model_type

    def on_epoch_end(self, epoch, logs=None):
        compute_latents_and_scores(self.model, self.X, self.metadata, self.labels_list, self.output_dir, epoch, self.sample_size,self.batch_size,self.model_type)

# Usage
# model_callback = ComputeLatentsCallback(model, X_train, metadata_df, ['batch', 'bio'], '/path/to/output')
# model.fit(X_train, y_train, epochs=10, callbacks=[model_callback])
