
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation,BatchNormalization
import collections
try: 
    from models.random_effects import *
except: 
    from random_effects import *

"""
Author: Aixa Andrade with collaboration of Son Nguyen.
Code inspired in the original ARMED's convolutional autoencoder code written by Kevin Nguyen for the melanoma experiment.
Some snippets of code are borrowed as they are from Kevin Nguyen et al 2023 (ARMED paper (2023)).
This code uses custom Dense layers for building custom ARMED vector autoencoders.

Chatgpt 4 was used to write docstrings.

"""


#we do not specify input dimension
#inspired in Medium article of Building an autoencoder with tied weights in keras by Laurence Mayrand-Provencher (2019)
#https://medium.com/@lmayrandprovencher/building-an-autoencoder-with-tied-weights-in-keras-c4a559c529a2

class TiedDenseTranspose(tf.keras.layers.Layer):
    """
    A tied dense transpose layer that shares weights with a source dense layer.

    Attributes:
        source_layer (tf.keras.layers.Dense): Source dense layer to tie weights with.
        activation (tf.keras.activations): Activation function for the layer.
        units (int): Number of units for the layer.
        kernel (tf.Variable): Shared weights with the source layer.
        bias_t (tf.Variable): Bias for the layer.
    """
    def __init__(self, source_layer: tf.keras.layers.Dense, units, activation=None, **kwargs):
        """
        Initialize the TiedDenseTranspose layer.

        Args:
            source_layer (tf.keras.layers.Dense): Source dense layer to tie weights with.
            units (int): Number of units for the layer.
            activation (str, optional): Activation function to use. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        self.source_layer = source_layer
        self.activation = tf.keras.activations.get(activation)
        self.units = units

        super().__init__(**kwargs)
        
    def build(self, batch_input_shape):
        """Build the layer weights."""
                
        # it only shares weights but not biases

        self.kernel = self.source_layer.kernel
        # initializes bias as zeros
        self.bias_t = self.add_weight(name='bias_t',
                                      shape=(self.units,),
                                      initializer="zeros")
        super().build(batch_input_shape)
    
    def call(self, inputs):
        """Apply the layer operations on the input tensor."""
        return self.activation(tf.matmul(inputs, self.kernel, transpose_b=True) + self.bias_t)

class Encoder(tf.keras.Model):
    """
    Encoder Layer for Neural Networks with optional batch normalization.

    Attributes:
        n_latent_dim (int): Number of latent dimensions.
        layer_units (list): List of units for each dense layer.
        return_layer_activations (bool): Flag to determine if layer activations should be returned.
        return_encoder_layers (bool): Flag to determine if encoder layers should be returned.
        layers (dict): Dictionary containing all the layers.
        dense_blocks (dict): Dictionary containing dense layers and batch normalization layers.
    """
    def __init__(self,
                 n_latent_dims: int=2, 
                 layer_units: list=[9, 7, 5], 
                 return_layer_activations: bool=False,
                 return_encoder_layers: bool=False,
                 use_batch_norm: bool=False, # Flag to determine if batch normalization should be used
                 name='encoder', 
                 **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.n_latent_dim = n_latent_dims        
        self.layer_units = layer_units
        self.return_layer_activations = return_layer_activations
        self.return_encoder_layers = return_encoder_layers
        self.use_batch_norm = use_batch_norm

        # Create dictionaries for the blocks and layers
        self.dense_blocks = {}
        self.all_layers = {}

        # Fill the dictionaries using a for loop
        for i, n_units in enumerate(self.layer_units):
            key_name = "dense_" + str(i)
            dense_layer = Dense(units=n_units, activation=None, name=key_name)  # activation is None if batch norm is used
            self.dense_blocks[key_name] = dense_layer
            self.all_layers[key_name] = [dense_layer]

            if self.use_batch_norm:
                bn_key_name = "batch_norm_" + str(i)
                bn_layer = BatchNormalization(name=bn_key_name)
                self.all_layers[key_name].append(bn_layer)

            # Add activation layer separately if using batch norm
            activation_layer = tf.keras.layers.Activation('selu')
            self.all_layers[key_name].append(activation_layer)

        # Define the latent layer
        self.dense_latent = Dense(units=self.n_latent_dim, activation="selu", name="dense_latent")
        self.all_layers["dense_latent"] = [self.dense_latent]

    def call(self, inputs, training=None):
        x = inputs
        layer_activations = []

        # Iterate through the layers using the dictionary
        for key, layers in self.all_layers.items():
            for layer in layers:
                x = layer(x, training=training)  # ensure to pass training parameter for batch normalization
            layer_activations.append(x)

        if self.return_layer_activations:
            return layer_activations
        elif self.return_encoder_layers:
            return self.all_layers, x
        else:
            return x

class Encoder_old(tf.keras.Model):
    """
    Encoder Layer for Neural Networks.

    Attributes:
        n_latent_dim (int): Number of latent dimensions.
        layer_units (list): List of units for each dense layer.
        return_layer_activations (bool): Flag to determine if layer activations should be returned.
        return_encoder_layers (bool): Flag to determine if encoder layers should be returned.
        layers (dict): Dictionary containing all the layers.
        dense_blocks (dict): Dictionary containing dense layers.
        """
    def __init__(self,
                 n_latent_dims: int=2, 
                 layer_units: list=[9,7,5], 
                 return_layer_activations: bool=False,
                 return_encoder_layers: bool=False,
                 name='encoder', 
                 **kwargs):
        """
        Initialize the Encoder.

        Args:
            n_latent_dims (int, optional): Number of latent dimensions. Defaults to 64.
            layer_units (list, optional): List containing the number of units for each dense layer. Defaults to [784, 392].
            return_layer_activations (bool, optional): Whether to return activations of each layer. Defaults to False.
            return_encoder_layers (bool, optional): Whether to return the encoder layers. Defaults to False.
            name (str, optional): Name of the layer. Defaults to 'encoder'.
            **kwargs: Additional keyword arguments.
        """


        super(Encoder, self).__init__(name=name, **kwargs)

        self.n_latent_dim = n_latent_dims        
        self.layer_units = layer_units
        self.return_layer_activations = return_layer_activations
        self.return_encoder_layers = return_encoder_layers

        # Create dictionaries for the blocks and layers
        self.dense_blocks = {}
        self.all_layers = {}

        # Fill the dictionaries using a for loop
        for i, n_units in enumerate(self.layer_units):
            key_name = "dense_" + str(i)
            self.dense_blocks[key_name] = Dense(units=n_units, activation="selu", name=key_name)

        # Define the latent layer
        self.dense_latent = Dense(units=self.n_latent_dim, activation="selu", name="dense_latent")

        # Store all layers in the layers dictionary
        self.all_layers = {**self.dense_blocks, "dense_latent": self.dense_latent}

    def call(self, inputs, training=None):
        """
        Call the encoder layer with input data.

        Args:
            inputs (tf.Tensor): Input tensor data.
            training (bool, optional): If in training mode or not. Defaults to None.

        Returns:
            tf.Tensor or list: Depending on the flags set during initialization, it either returns:
                - Activations of each layer
                - Encoder layers and the final output
                - Only the final output
        """

        x = inputs
        layer_activations = []

        # Iterate through the layers using the dictionary
        for key, layer in self.all_layers.items():
            x = layer(x)
            layer_activations.append(x)

        if self.return_layer_activations:
            return layer_activations
        elif self.return_encoder_layers:
            return self.all_layers, x
        else:
            return x

class Decoder(tf.keras.Model):
    """
    Decoder Layer for Neural Networks. 
    The Decoder layers can be Tied with the Encoder layers if encoder layers are provided and if the tied_weights = True.

    Attributes:
        encoder_dense_layers (list): List of encoder dense layers to tie weights with.
        in_shape (tuple): Input shape for the autoencoder (encoder input shape). Input shape encoder = output shape decoder
        layer_units (list): List of units for each dense layer.
        last_activation (str): Last activation function for the decoder.
        layers (dict): Dictionary containing all the layers.
    """
    def __init__(self,
                in_shape: tuple,
                encoder_layers: list = [],              
                layer_units: list=[9,7,5],
                last_activation: str='sigmoid',
                name='decoder',
                tied_weights = True, 
                **kwargs):
        """
        Initialize the Decoder.

        Args:
            encoder_layers (list, optional): List of encoder layers to tie weights with. If you want a TiedDecoder, you have to provide the encoder_layers. Defaults to empty list.
            in_shape (tuple):  Input shape for the autoencoder (encoder input shape). Input shape encoder = output shape decoder
            layer_units (list, optional): List containing the number of units for each dense layer. Defaults to [784, 392].
            last_activation (str, optional): Last activation function for the decoder. Defaults to "sigmoid".
            name (str, optional): Name of the layer. Defaults to 'decoder'.
            tied_weights (bool, optional): If True, the layers of the Decoder are Tied with the Encoder. Else: The layers of the Decoder are Dense. Defaults to True.
            **kwargs: Additional keyword arguments.
        """
        super(Decoder, self).__init__(name=name, **kwargs)

        self.in_shape = in_shape
        self.layer_units = layer_units
        self.last_activation = last_activation

        self.all_layers = {}

        self.tied_weights = tied_weights

        if (self.tied_weights == True )& (len(encoder_layers)>0):  
            #If tied weights = True --> decoder layers are tied with the encoder layers 
            #print("encoder layers",encoder_layers)
            # get encoder dense layers
            # encoder_dense_layers = [layer for layer in encoder_layers if "dense" in layer.name]
            def is_iterable(obj):
                """ Check if the object is iterable but not a string """
                return isinstance(obj, collections.abc.Iterable) and not isinstance(obj, (str, bytes))

            # Using a nested list comprehension to handle both nested and flat list scenarios
            encoder_dense_layers = [layer for item in encoder_layers
                                    for layer in (item if is_iterable(item) else [item])
                                    if "dense" in layer.name]

            self.encoder_dense_layers = encoder_dense_layers    
            # build the decoder reverse looping through the encoder layers
            for n_units, e_layer in zip(self.layer_units[::-1], self.encoder_dense_layers[1:][::-1]):
                key_name = e_layer.name + "_t"
                self.all_layers[key_name] = TiedDenseTranspose(source_layer=e_layer, units=n_units, activation="selu", name=key_name)

            # out decoder: out layer shares weights with encoder first layer
            # defining layer with last activation      
            # the last activation is sigmoid to make sure the values are between zero and one
            key_name = "dense_out"
            self.all_layers[key_name] = TiedDenseTranspose(source_layer=self.encoder_dense_layers[0], units=self.in_shape[-1], activation=self.last_activation, name=key_name)

        else:
            #If tied weights = False --> decoder layers are Dense layers
            # build the decoder reverse looping through the layer units
            for i,n_units in enumerate(self.layer_units[::-1]):
                key_name = "dense_"+str(len(self.layer_units)-i)
                self.all_layers[key_name] = Dense(units=n_units, activation="selu", name=key_name)
    
            # the last activation is sigmoid to make sure the values are between zero and one
            key_name = "dense_out"
            self.all_layers[key_name] = Dense(units=self.in_shape[-1], activation=self.last_activation, name=key_name)

    def call(self, inputs, training=None):
        """
        Call the decoder layer with input data.

        Args:
            inputs (tf.Tensor): Input tensor data.
            training (bool, optional): If in training mode or not. Defaults to None.

        Returns:
            tf.Tensor: Processed output tensor.
        """
        x = inputs
        # apply transposed dense layers (decoder)
        for key, layer in self.all_layers.items():
            #print(layer.name)
            #x = layer(x)
            x = layer(x, training=training)
        return x



class AE(tf.keras.Model):
    """
    Autoencoder (AE) Model with tied weights.

    Attributes:
        in_shape (tuple): Input shape for the AE.
        layer_units (list): List of units for each dense layer in the encoder.
        n_latent_dims (int): Number of latent dimensions for the encoder.
        last_activation (str): Last activation function for the decoder.
        return_layer_activations (bool): Whether to return layer activations from the encoder.
        encoder (Encoder): Encoder part of the AE.
        decoder (Decoder): Decoder part of the AE, it has Tied weights with the Encoder.
    """
    
    def __init__(self, 
                 in_shape: tuple,
                 n_latent_dims: int = 2, 
                 layer_units: list = [9,7,5], 
                 last_activation: str = "sigmoid",
                 return_layer_activations: bool = False,
                 use_batch_norm: bool=False,
                 name='ae', 
                 **kwargs):
        """
        Initialize the AE model.

        Args:
            in_shape (tuple): Input shape for the AE.
            n_latent_dims (int, optional): Number of latent dimensions for the encoder. Defaults to 64.
            layer_units (list, optional): List containing the number of units for each dense layer in the encoder. Defaults to [784, 392].
            last_activation (str, optional): Last activation function for the decoder. Defaults to "sigmoid".
            return_layer_activations (bool, optional): Whether to return layer activations from the encoder. Defaults to False.
            name (str, optional): Name of the model. Defaults to 'ae'.
            **kwargs: Additional keyword arguments.
        """
        super(AE, self).__init__(name=name, **kwargs)

        self.in_shape = in_shape
        self.layer_units = layer_units
        self.n_latent_dims = n_latent_dims
        self.last_activation = last_activation
        self.return_layer_activations = return_layer_activations
        self.use_batch_norm = use_batch_norm

        self.encoder = Encoder(n_latent_dims=n_latent_dims, 
                               layer_units=layer_units,
                               return_layer_activations=self.return_layer_activations,
                               use_batch_norm=self.use_batch_norm)
        
        # Assuming the Encoder class returns a dictionary for its layers attribute
        encoder_layers_list = list(self.encoder.all_layers.values())
        self.decoder = Decoder(in_shape=self.in_shape, encoder_layers=encoder_layers_list,
                               layer_units=self.layer_units, 
                               last_activation=self.last_activation)

    def call(self, inputs, training=None):
        """
        Call the AE model with input data.

        Args:
            inputs (tf.Tensor): Input tensor data.
            training (bool, optional): If in training mode or not. Defaults to None.

        Returns:
            tf.Tensor: Processed output tensor.
        """
        # Get the encoder output. If return_layer_activations is True, 
        # the encoder returns all layer activations, else just the latent representation.
        encoder_output = self.encoder(inputs, training=training)

        # Determine the latent representation based on return_layer_activations flag
        latent = encoder_output[-1] if self.return_layer_activations else encoder_output
        
        out = self.decoder(latent, training=training)
        return out

class AEC(tf.keras.Model):
    """
    An autoencoder-based classifier model built using TensorFlow's Keras API.

    This model is a combination of an autoencoder for unsupervised learning and a classifier for supervised learning. It is designed to work with input data in the specified input shape, compress it into a latent space using an encoder, and then reconstruct the input from this compressed representation using a decoder. Additionally, it uses the latent representation for classification purposes.

    The autoencoder part of the model is a 'tied' autoencoder, meaning that the weights of the encoder are tied to the weights of the decoder. This type of architecture can be beneficial for certain types of data compression and reconstruction tasks.

    Parameters:
    in_shape (tuple): The shape of the input data.
    n_latent_dims (int, optional): The number of dimensions for the latent space representation. Default is 2.
    layer_units (list, optional): The number of units in each layer of the encoder (and by extension, the decoder). Default is [9, 7, 5].
    last_activation (str, optional): The activation function to be used in the last layer of the autoencoder. Default is 'sigmoid'.
    return_layer_activations (bool, optional): Flag to determine whether the encoder should return all layer activations or just the final latent representation. Default is False.
    n_pred (int, optional): The number of prediction classes for the classifier. Default is 20.
    layer_units_latent_classifier (list, optional): The number of units in each layer of the classifier. Default is [2].
    name (str, optional): Name of the model. Default is 'ae_class'.

    The model has three main components:
    - An encoder that reduces the input to a lower-dimensional latent space.
    - A decoder that reconstructs the input from the latent space.
    - A classifier that uses the latent space representation for classification tasks.

    The `call` method of the model takes in input data and optionally a training flag and returns a dictionary with two keys: 'reconstruction_output' for the output of the autoencoder, and 'classification_output' for the output of the classifier.

    Example:
        model = AE_classifier(in_shape=(28, 28, 1))
        # For training or inference
        output = model(data)
    """

    
    def __init__(self, 
                 in_shape: tuple,
                 n_latent_dims: int = 2, 
                 layer_units: list = [9,7,5], 
                 last_activation: str = "sigmoid",
                 return_layer_activations = False,
                 n_pred=20,
                 layer_units_latent_classifier=[2],
                 use_batch_norm: bool=False,
                 name='aec', 
                 **kwargs):

        super(AEC, self).__init__(name=name, **kwargs)

        self.in_shape = in_shape
        self.layer_units = layer_units
        self.n_latent_dims = n_latent_dims
        self.last_activation = last_activation
        self.return_layer_activations = return_layer_activations
        self.n_pred =  n_pred
        self.layer_units_latent_classifier = layer_units_latent_classifier
        self.use_batch_norm = use_batch_norm

        self.encoder = Encoder(n_latent_dims=n_latent_dims, 
                               layer_units=layer_units,
                               return_layer_activations=self.return_layer_activations,
                               use_batch_norm=self.use_batch_norm)
        
        # Assuming the Encoder class returns a dictionary for its layers attribute
        encoder_layers_list = list(self.encoder.all_layers.values())
        # Tied AE. ENCODER WEIGHTS = DECODER
        self.decoder = Decoder(in_shape=self.in_shape, encoder_layers=encoder_layers_list,
                               layer_units=self.layer_units, 
                               last_activation=self.last_activation)






        self.classifier = Classifier(n_clusters=self.n_pred,layer_units = self.layer_units_latent_classifier)

    def call(self, inputs, training=None):
        # Get the encoder output. If return_layer_activations is True, 
        # the encoder returns all layer activations, else just the latent representation.
        #print("model inputs shape",inputs.shape)
        encoder_output = self.encoder(inputs, training=training)

        # Determine the latent representation based on return_layer_activations flag
        latent = encoder_output[-1] if self.return_layer_activations else encoder_output

        # Pass the latent representation through the decoder and classifier
        recon = self.decoder(latent, training=training)
        classification = self.classifier(latent)

        #print("recon pred shape",recon.shape,"class pred shape",classification.shape)
    
        return {'reconstruction_output': recon, 'classification_output': classification}



class AdversarialClassifier(tkl.Layer):
    def __init__(self,
                 n_clusters: int, 
                 n_latent_dims: int=2,
                 layer_units: list=[5, 4],
                 name: str='adversary',
                 **kwargs):
        """Adversarial classifier. 

        Args:
            n_clusters (int): number of clusters (classes)
            layer_units (list, optional): Neurons in each layer. Can be a list of any
                length. Defaults to [8, 8, 8].
            name (str, optional): Model name. Defaults to 'adversary'.
        """        
        
        super(AdversarialClassifier, self).__init__(name=name, **kwargs)
        
        self.n_clusters = n_clusters
        self.layer_units = layer_units
        
        self.all_layers = []
        for iLayer, neurons in enumerate(layer_units):
            self.all_layers += [tkl.Dense(neurons, 
                                      activation='relu', 
                                      name=name + '_dense' + str(iLayer))]
            
        self.all_layers += [tkl.Dense(self.n_clusters , activation='softmax', name=name + '_dense_out')]
        
    def call(self, inputs):
        if type(inputs) is list:
            inputs = tf.concat(inputs, axis=-1)
        x = inputs
        for layer in self.all_layers:
            x = layer(x)
            
        return x
    
    def get_config(self):
        return {'n_clusters': self.n_clusters,
                'layer_units': self.layer_units}


class Adversarial(tf.keras.layers.Layer):
    """
    Adversarial Classifier Layer. Based on Conv layers adv (OLD VERSION. This is not currently being used for the Domain adv classifier. Adversarial classifier is the version that is being used).
    
    An adversarial classifier layer that classifies a given input into 
    specified number of clusters. It's designed to be used in conjunction
    with other neural network architectures to introduce an adversarial 
    classification task.

    Attributes:
        n_clusters (int): Number of clusters/classes for classification.
        n_latent_dims (int): Number of latent dimensions.
        layer_units (list): List of units for each dense layer.
        adv_layer_units (list): List of units for each adversarial dense layer.
        adv_blocks (list): List containing sets of dense, concatenate, and ReLU layers.
        dense_out (tf.keras.layers.Dense): Dense output layer.
        softmax (tf.keras.layers.Dense): Softmax layer for classification.
    """
    
    def __init__(self,
                n_clusters: int, 
                n_latent_dims: int=2,
                layer_units: list=[5, 4], 
                name='adversary', 
                **kwargs):
        """
        Initialize the Adversarial Classifier layer.

        Args:
            n_clusters (int): Number of clusters/classes for classification.
            n_latent_dims (int, optional): Number of latent dimensions. Defaults to 64.
            layer_units (list, optional): List containing the number of units for each dense layer. Defaults to [784, 392].
            name (str, optional): Name of the layer. Defaults to 'adversary'.
            **kwargs: Additional keyword arguments.
        """

        super(Adversarial, self).__init__(name=name, **kwargs)     

        self.n_clusters = n_clusters
        self.n_latent_dims = n_latent_dims
        self.layer_units = layer_units
        self.adv_layer_units = self.layer_units[1:] + [self.n_latent_dims]
        self.adv_blocks = []

        # Fill the blocks with a for loop
        for i, n_units in enumerate(self.adv_layer_units):
            dense_i = Dense(units=n_units, name="dense_" + str(i))
            concat_i = tf.keras.layers.Concatenate(axis=-1, name=name + 'concat_' + str(i))
            act_i = tf.keras.layers.ReLU(name=name + 'relu_' + str(i))
            self.adv_blocks += [(dense_i, concat_i, act_i)] 

        self.dense_out = Dense(units=self.n_latent_dims, name=name + 'dense_out')
        self.softmax = Dense(units=self.n_clusters, activation='softmax', name=name + '_softmax')

    def call(self, inputs, training=None):
        """
        Call the Adversarial Classifier layer with input data.

        Args:
            inputs (list of tf.Tensor): List containing input tensor data and auxiliary tensors.
            training (bool, optional): If in training mode or not. Defaults to None.

        Returns:
            tf.Tensor: Processed output tensor.
        """
        
        # Activation zero
        x = inputs[0]
        # Other activations
        a = inputs[1:]
        
        # Process through adversarial blocks
        for a_i, adv_bloc_i in zip(a, self.adv_blocks):
            # Get layers i
            dense_i, concat_i, act_i = adv_bloc_i
            # Apply layers from adv block i
            x = dense_i(x)
            x = concat_i([x, a_i])
            x = act_i(x)

        x = self.dense_out(x)
        x = self.softmax(x)
        return x

class Classifier(tf.keras.layers.Layer):
    """
        A custom Keras Layer for classification tasks.

        This layer implements a classifier with a user-defined number of dense layers 
        followed by an output layer for clustering (can be used to predict donors/batches). Optionally, it can also implement
        another classifier subnet for a second prediction (can be used to predict celltypes).

        Attributes:
        -----------
        layer_units : list
            List of integers specifying the number of units in each dense layer.
        layers_cluster : dict
            Dictionary containing dense layers for clustering.
        n_clusters : int
            Number of clusters for the classification task.
        n_pred : int
            Number of predictions.
        get_pred : bool
            Flag to determine if prediction subnet should be built.
        layers_pred : dict
            Dictionary containing dense layers for prediction, built only if get_pred is True.

        Methods:
        --------
        call(inputs, training=None):
            Perform the forward pass for the clustering and optionally for prediction.

        get_config():
            Returns a dictionary containing the configuration of the classifier (i.e., n_clusters).

        Parameters:
        -----------
        n_clusters : int
            Number of clusters for the classification task.
        layer_units : list, optional
            List of integers specifying the number of units in each dense layer. Defaults to [32, 16].
        n_pred : int, optional
            Number of predictions, only used if get_pred is True. Defaults to 4.
        get_pred : bool, optional
            Flag to determine if a subnet for predictions should be built. Defaults to False.
        name : str, optional
            Name of the layer. Defaults to 'latent_classifier'.
        **kwargs : 
            Additional keyword arguments inherited from tf.keras.layers.Layer.

    """

    def __init__(self, 
                 n_clusters: int,
                 layer_units: list=[2],
                 n_pred: int = 10,
                 get_pred = False,
                 name='latent_classifier', 
                 **kwargs):

        super(Classifier, self).__init__(name=name, **kwargs)

        self.layer_units = layer_units
        self.layers_cluster = {}
        self.n_clusters = n_clusters 
        self.n_pred = n_pred
        self.get_pred = get_pred
        if self.get_pred:
            self.layers_pred = {}
        

        # Fill the dictionaries using a for loop
        for i, n_units in enumerate(self.layer_units):
            key_name = "dense_" + str(i)
            self.layers_cluster[key_name] = Dense(units=n_units, activation="relu", name=key_name)
            # if get_pred ==True, build dense subnet to get_predictions
            if self.get_pred:
                self.layers_pred[key_name] = Dense(units=n_units, activation="relu", name=key_name)
        
        #This layer predicts the number of clusters
        self.layers_cluster["dense_out"] = Dense(self.n_clusters, activation='softmax', name=name + '_out')
        if self.get_pred:
            #if get_pred ==True: use softmax to pred the classes
            self.layers_pred["dense_out"] = Dense(self.n_pred, activation='softmax', name=name + '_out')
      
    def call(self, inputs, training=None):
        c = inputs
        for key, layer in self.layers_cluster.items():
            c = layer(c)
            # final c: vector of n samples * n clusters with the probability of each sample being of each cluster
        if self.get_pred:
            y = inputs
            for key, layer in self.layers_pred.items():
                y = layer(y)
            #return class predictions, cluster predictions
            return y,c 
        else:
            return c
        
    def get_config(self):
        return {'n_clusters': self.n_clusters}



class DomainAdversarialAE(AE):
    """
    Domain Adversarial Autoencoder (DAAE) Class.
    
    An extension of the autoencoder (AE) that integrates an adversarial 
    classifier in its architecture to perform unsupervised domain adaptation.
    This class enables training of the AE such that the latent representation 
    is invariant to domain shifts, making the model robust against changes in 
    the data distribution.

    Attributes:
        in_shape (tuple): Input shape of the data.
        n_clusters (int): Number of clusters for adversarial classification.
        n_latent_dims (int): Dimensions of the latent space.
        layer_units (list): List containing units for each dense layer.
        last_activation (str): Activation function for the last layer.
        encoder (Encoder): Encoder part of the DAAE.
        decoder (Decoder): Decoder part of the DAAE.
        adversary (AdversarialClassifier): Adversarial classifier for domain adaptation.
    """

    def __init__(self, in_shape: tuple, n_clusters: int, 
                 n_latent_dims: int=2, 
                 layer_units: list=[9,7,5],
                 last_activation: str="sigmoid",
                 n_pred: int=10,
                 layer_units_latent_classifier: list=[2],
                 get_pred=False,
                 use_batch_norm: bool=False,
                 name='da_ae', 
                 **kwargs):
        """
        Initialize the Domain Adversarial Autoencoder.

        Args:
            in_shape (tuple): Input shape of the data.
            n_clusters (int): Number of clusters for adversarial classification.
            n_latent_dims (int, optional): Dimensions of the latent space. Defaults to 64.
            layer_units (list, optional): List containing units for each dense layer. Defaults to [784, 392].
            last_activation (str, optional): Activation function for the last layer. Defaults to "sigmoid".
            name (str, optional): Name of the model. Defaults to 'da_ae'.
            **kwargs: Additional keyword arguments.
        """

        super(AE, self).__init__(name=name, **kwargs)

        self.in_shape = in_shape 
        self.n_clusters = n_clusters 
        self.n_latent_dims = n_latent_dims 
        self.layer_units = layer_units
        self.last_activation = last_activation
        self.get_pred = get_pred
        self.use_batch_norm = use_batch_norm
                
        if self.get_pred:
            self.n_pred = n_pred
            self.layer_units_latent_classifier = layer_units_latent_classifier
            #The latent classifier returns class predictions 
            self.latent_classifier = Classifier(n_clusters=self.n_pred,layer_units = self.layer_units_latent_classifier)
        
        #autoencoder: encoder +decoder
        self.encoder = Encoder(n_latent_dims = n_latent_dims,
                                 layer_units=self.layer_units,
                                 return_layer_activations=True,
                                 use_batch_norm=self.use_batch_norm)
        encoder_layers_list = list(self.encoder.all_layers.values())
        self.decoder = Decoder(in_shape=self.in_shape,encoder_layers = encoder_layers_list,layer_units = self.layer_units, last_activation = self.last_activation)
        #adversarial classifier
        self.adversary = AdversarialClassifier(n_clusters = self.n_clusters,
                                                 n_latent_dims = self.n_latent_dims,
                                                 layer_units=self.layer_units)
    
    def call(self, inputs,training=None):

        """
        Forward pass through the Domain Adversarial Autoencoder.

        Args:
            inputs (tuple): Tuple containing the input data and cluster information.

        Returns:
            tuple: Reconstruction from the decoder and prediction from the adversarial classifier.
        """

        x, clusters = inputs
        #print(x.shape)
        # encoder
        encoder_activations = self.encoder(x,training=training)
        # apply adversary to encoder activations (decoder shares weights with encoder)
        pred_cluster = self.adversary(encoder_activations)
        # latent space is the last activation layer
        latent = encoder_activations[-1]
        # decoder is applied to latent
        recon = self.decoder(latent,training=training)

        if self.get_pred:
            # classification
            pred_class = self.latent_classifier(latent)
            return (recon, pred_class, pred_cluster)
        else:
            return (recon, pred_cluster)
        

    def compile(self,
                loss_recon=tf.keras.losses.MeanSquaredError(),
                loss_multiclass=tf.keras.losses.CategoricalCrossentropy(),
                metric_multiclass=tf.keras.metrics.CategoricalAccuracy(name='acc'),
                opt_autoencoder=tf.keras.optimizers.Adam(lr=0.0001),
                opt_adversary=tf.keras.optimizers.Adam(lr=0.0001),
                loss_recon_weight=1.0,
                loss_gen_weight=0.05,
                loss_class_weight=0.01):
        """
        Compile the model with specified losses, metrics, and optimizers.

        Args:
            loss_recon (tf loss): Reconstruction loss function.
            loss_multiclass (tf loss): multiclass loss function. It works for all multiclass tasks.
            metric_multiclass (tf metric): Metric for adversarial classifier performance.
            opt_autoencoder (tf optimizer): Optimizer for autoencoder.
            opt_adversary (tf optimizer): Optimizer for adversarial classifier.
            loss_recon_weight (float): Weight for the reconstruction loss.
            loss_gen_weight (float): Weight for the adversarial loss (generator part).
            loss_class_weight (float): Weight for the class loss. Only used if get_pred ==True.
        """


        super().compile()

        self.loss_recon = loss_recon
        # adv and class are the same loss but I decided to use diff names
        self.loss_adv = loss_multiclass
        self.loss_class = loss_multiclass

        self.opt_autoencoder = opt_autoencoder
        self.opt_adversary = opt_adversary
        
        # track mean loss
        self.loss_recon_tracker = tf.keras.metrics.Mean(name='recon_loss')
        self.loss_adv_tracker = tf.keras.metrics.Mean(name='adv_loss')
        self.loss_total_tracker = tf.keras.metrics.Mean(name='total_loss')

        # define metrics
        self.metric_adv = metric_multiclass
        self.metric_class = metric_multiclass

        # define loss weights
        self.loss_recon_weight = loss_recon_weight
        self.loss_gen_weight = loss_gen_weight

        if self.get_pred: # define latent class loss, metric and weights
            self.metric_multiclass = metric_multiclass
            self.loss_class_weight = loss_class_weight
            self.loss_class_tracker = tf.keras.metrics.Mean(name='class_loss')
    @property
    def metrics(self):
        if self.get_pred:
            return [self.loss_recon_tracker,
                self.loss_class_tracker,
                self.loss_adv_tracker,
                self.loss_total_tracker,
                self.metric_adv,
                self.metric_class]
        else:
            return [self.loss_recon_tracker,
                self.loss_adv_tracker,
                self.loss_total_tracker,
                self.metric_adv]
        
    def train_step(self, data):
        """
        Perform a training step for the model.

        Args:
            data (tuple): Tuple containing input data, target data, and optionally sample weights.

        Returns:
            dict: Dictionary containing values of tracked metrics.
        """

        #load data
        x, clusters = data[0]

        if self.get_pred:
            _, labels = data[1]
        # else:
        #     labels = None

        sample_weights = None if len(data) != 3 else data[2]

        #CHECK IF THE SHAPES ARE CORRECT
        assert x.shape[0] == clusters.shape[0], "Mismatch between x and clusters"
        if self.get_pred:
            assert x.shape[0] == labels.shape[0], "Mismatch between x and labels"

        #Train adversary
        encoder_outs = self.encoder(x, training=True)
        #calculate adv loss
        with tf.GradientTape() as gt:
            pred_cluster = self.adversary(encoder_outs)
            loss_adv = self.loss_adv(clusters, pred_cluster, sample_weight=sample_weights)
        
        #apply gradients
        grads_adv = gt.gradient(loss_adv, self.adversary.trainable_variables)
        # minimizing adv loss (remove comments)
        self.opt_adversary.apply_gradients(zip(grads_adv, self.adversary.trainable_variables))
        
        # Update adversarial loss tracker
        self.metric_adv.update_state(clusters, pred_cluster)
        self.loss_adv_tracker.update_state(loss_adv)

        # Train autoencoder
        with tf.GradientTape(persistent=True) as gt2:
            #apply model   
            outputs = self(inputs=(x, clusters), training=True)
            if self.get_pred:
                pred_recon, pred_class, pred_cluster = outputs #(+ pred class)
            else:
                pred_recon, pred_cluster = outputs
            
            #compute individual losses
            loss_recon = self.loss_recon(x, pred_recon, sample_weight=sample_weights)
            loss_adv = self.loss_adv(clusters, pred_cluster, sample_weight=sample_weights)
            if self.get_pred:
                loss_class = self.loss_class(labels, pred_class, sample_weight=sample_weights)
                #add class loss to total loss: (recon) - adv loss (gen) +class loss
                total_loss = (self.loss_recon_weight * loss_recon) \
                    + (self.loss_class_weight * loss_class) \
                    - (self.loss_gen_weight * loss_adv) 
            else:
                #compute total ae loss: (recon) - adv loss (gen)
                total_loss = (self.loss_recon_weight * loss_recon)- (self.loss_gen_weight * loss_adv)

        if self.get_pred: # +latent classifier trainable vars
                lsWeights = self.encoder.trainable_variables + self.decoder.trainable_variables \
                + self.latent_classifier.trainable_variables
        else:
            lsWeights = self.encoder.trainable_variables + self.decoder.trainable_variables
        
        #backpropagate
        grads_aec = gt2.gradient(total_loss, lsWeights)
        self.opt_autoencoder.apply_gradients(zip(grads_aec, lsWeights))

        # Update loss trackers
        if self.get_pred:
            self.metric_class.update_state(labels, pred_class)
            self.loss_class_tracker.update_state(loss_class)
        self.loss_recon_tracker.update_state(loss_recon)
        self.loss_total_tracker.update_state(total_loss)
        
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):

        """
        Perform a testing (validation) step for the model.

        Args:
            data (tuple): Tuple containing input data and target data.

        Returns:
            dict: Dictionary containing values of tracked metrics.
        """

        x, clusters = data[0]
        if self.get_pred:
            _, labels = data[1]
        # else:
        #     labels = None

        #CHECK IF THE SHAPES ARE CORRECT
        assert x.shape[0] == clusters.shape[0], "Mismatch between x and clusters"
        if self.get_pred:
            assert x.shape[0] == labels.shape[0], "Mismatch between x and labels"

        # apply model     
        outputs = self(inputs=(x, clusters), training=False)
        if self.get_pred:
            pred_recon, pred_class, pred_cluster = outputs #(+ pred class)
        else:
            pred_recon, pred_cluster = outputs
       

        # compute ind losses
        loss_recon = self.loss_recon(x, pred_recon)
        loss_adv = self.loss_adv(clusters, pred_cluster)

        #compute total loss
        if self.get_pred:
            loss_class = self.loss_class(labels, pred_class)
                #add class loss to total loss: (recon) - adv loss (gen) +class loss
            total_loss = (self.loss_recon_weight * loss_recon) \
                    + (self.loss_class_weight * loss_class) \
                    - (self.loss_gen_weight * loss_adv)  
        else:     
            total_loss = (self.loss_recon_weight * loss_recon)- (self.loss_gen_weight * loss_adv)
                    
        #update metrics and losses
        self.metric_adv.update_state(clusters, pred_cluster)
        self.loss_recon_tracker.update_state(loss_recon)
        self.loss_adv_tracker.update_state(loss_adv)
        self.loss_total_tracker.update_state(total_loss)
        if self.get_pred:
            self.metric_class.update_state(labels, pred_class)
            self.loss_class_tracker.update_state(loss_class)
        
        return {m.name: m.result() for m in self.metrics}

class RandomEffectEncoder(Encoder):
    """
    RandomEffectEncoder: A specialized encoder that incorporates random effects with dense layers.
    
    Inherits from the provided Encoder class. This encoder is designed to model random effects by 
    introducing specialized layers for handling them. Each dense layer is followed by a random effect layer 
    and an activation layer.

    Attributes:
        n_latent_dims (int): Number of latent dimensions.
        layer_units (list): List containing the number of units for each dense layer.
        post_loc_init_scale (float): Initial scale for the location of the posterior distribution.
        prior_scale (float): Scale for the prior distribution.
        kl_weight (float): Weighting factor for the KullbackLeibler divergence.
        re_layers (dict): Dictionary containing random effect layers.
        act_layers (dict): Dictionary containing activation layers.
        layer_blocks (dict): Dictionary containing blocks of (dense, random effect, activation) layers.

    Args:
        n_latent_dims (int, optional): Number of latent dimensions. Defaults to 2.
        layer_units (list, optional): List containing the number of units for each dense layer. Defaults to [8].
        post_loc_init_scale (float, optional): Initial scale for the location of the posterior distribution. Defaults to 0.1.
        prior_scale (float, optional): Scale for the prior distribution. Defaults to 0.25.
        kl_weight (float, optional): Weighting factor for the KullbackLeibler divergence. Defaults to 1e-5.
        name (str, optional): Name of the encoder. Defaults to 'encoder'.
        **kwargs: Additional keyword arguments.
        
    """

    
    def __init__(self,
                n_latent_dims: int=2, 
                layer_units: list=[8],
                post_loc_init_scale: float=0.1,
                prior_scale: float=0.25,
                kl_weight: float=1e-5, 
                name = 'encoder',
                **kwargs):
        """ Initialize the RandomEffectEncoder. """



        super(RandomEffectEncoder, self).__init__(n_latent_dims=n_latent_dims, 
                                                  layer_units=layer_units,name=name, **kwargs)


        #dictionary of random effect layers
        self.re_layers = {}
        
        #dictionary of activation layers
        self.act_layers = {}

        #Build blocks of (dense, RE, activation layers)
        self.layer_blocks = {}
        #dense blocks are inherited from Encoder class
        
        for key, layer in self.dense_blocks.items():
            #layer i 
            layer_i = key.split("_")[-1]
            #random effect layer
            self.re_layers["re_"+layer_i] = ClusterScaleBiasBlock(layer.units,
                                                post_loc_init_scale = post_loc_init_scale,
                                                prior_scale = prior_scale,
                                                kl_weight = kl_weight,
                                                name = name + '_re_'+layer_i)

            #act layer
            self.act_layers["act_"+layer_i]  = Activation('selu')
            #add blocks of (dense, RE, activation layers)
            self.layer_blocks["block_"+layer_i] = (layer,self.re_layers["re_"+layer_i], self.act_layers["act_"+layer_i])
        #define re_encoder_layers
        self.re_encoder_layers = {**self.layer_blocks, "dense_latent": self.dense_latent}

    def call(self, inputs, training=None):

        """
        Forward pass for the RandomEffectEncoder.
        
        Args:
            inputs (tuple): A tuple containing two elements - the input data (x) and the random effects data (z).
            training (bool, optional): If in training mode or not. Defaults to None.

        Returns:
            tf.Tensor: Transformed input after passing through dense, random effect and activation layers.
        """

        x, z = inputs

        for key, (dense, re, activation) in self.layer_blocks.items():
            x = dense(x)
            x = re((x, z), training=training)
            x = activation(x)
        x = self.dense_latent(x)  
        return x
    # def summary(self):
    #     print("RandomEffectEncoder Summary:")
    #     print(f"{'Layer':<20} {'Output Shape':<20} {'# Params':<10}")
    #     for name, (dense, re, activation) in self.layer_blocks.items():
    #         # Assuming model has been built at least once so these methods can be accessed
    #         print(f"{dense.name:<20} {str(dense.output_shape):<20} {dense.count_params():<10}")
    #         print(f"{re.name:<20} {'-':<20} {re.count_params():<10}")
    #         print(f"{activation.name:<20} {'-':<20} {'-':<10}")
    #     print(f"{self.dense_latent.name:<20} {str(self.dense_latent.output_shape):<20} {self.dense_latent.count_params():<10}")


class RandomEffectDecoder(Decoder):
    def __init__(self,
                in_shape: tuple, 
                layer_units: list=[8],
                last_activation: str='sigmoid',
                post_loc_init_scale: float=0.1,
                prior_scale: float=0.25,
                kl_weight: float=1e-5, 
                name = 'decoder',
                **kwargs):
        """ Initialize the RandomEffectDecoder. """

        #I do not want tied weights in the RandomEffectDecoder
        super(RandomEffectDecoder, self).__init__(in_shape = in_shape, 
                                                  layer_units = layer_units,
                                                  last_activation = last_activation,
                                                  name = name,
                                                  tied_weights = False, 
                                                  **kwargs)

        #dictionary of random effect layers
        self.re_layers = {}
        
        #dictionary of activation layers
        self.act_layers = {}

        #Build blocks of (dense, RE, activation layers)
        self.layer_blocks = {}
        #dense blocks are inherited from Encoder class
        
        for key, layer in self.all_layers.items():
            #layer i 
            layer_i = key.split("_")[-1]
            #random effect layer
            self.re_layers["re_"+layer_i] = ClusterScaleBiasBlock(layer.units,
                                                post_loc_init_scale = post_loc_init_scale,
                                                prior_scale = prior_scale,
                                                kl_weight = kl_weight,
                                                name = name + '_re_'+layer_i)
           
            if key == 'dense_out': #for the block that has the dense_out layer, the activation layer = last_activation
                self.act_layers["last_act"]  = Activation(self.last_activation, name = name + '_act_'+self.last_activation)
                #add blocks of (dense, RE, activation layers)
                self.layer_blocks["block_"+layer_i] = (layer,self.re_layers["re_"+layer_i], self.act_layers["last_act"])

            else: #all other activation layers are 'relu'
                self.act_layers["act_"+layer_i]  = Activation('selu', name = name + '_act_'+layer_i)
            
                #add blocks of (dense, RE, activation layers)
                self.layer_blocks["block_"+layer_i] = (layer,self.re_layers["re_"+layer_i], self.act_layers["act_"+layer_i])

        self.re_decoder_layers = self.layer_blocks
        
    def call(self, inputs, training=None):

        """
        Forward pass for the RandomEffectDecoder.
        
        Args:
            inputs (tuple): A tuple containing two elements - the input data (x) and the random effects data (z).
            training (bool, optional): If in training mode or not. Defaults to None.

        Returns:
            tf.Tensor: Transformed input after passing through dense, random effect and activation layers.
        """

        x, z = inputs

        for key, (dense, re, activation) in self.layer_blocks.items():
            x = dense(x)
            x = re((x, z), training=training)
            x = activation(x)
        return x
    # def summary(self):
    #     print("RandomEffectDecoder Summary:")
    #     print(f"{'Layer':<20} {'Output Shape':<20} {'# Params':<10}")
    #     for name, (dense, re, activation) in self.layer_blocks.items():
    #         # Assuming model has been built at least once so these methods can be accessed
    #         print(f"{dense.name:<20} {str(dense.output_shape):<20} {dense.count_params():<10}")
    #         print(f"{re.name:<20} {'-':<20} {re.count_params():<10}")
    #         print(f"{activation.name:<20} {'-':<20} {'-':<10}")

class DomainEnhancingAutoencoderClassifier(tf.keras.Model):
    """
    Domain-enhanced autoencoder model for classification and clustering.

    This model leverages an autoencoder structure with a domain-enhanced approach to perform 
    classification and clustering tasks. It comprises an encoder (`RandomEffectEncoder`), a decoder 
    (`RandomEffectDecoder`), and a classifier which operates in both the latent and reconstructed space. 
    The model can predict clusters or class labels based on the latent and reconstructed representations.

    Parameters:
    ------------
    - in_shape (tuple): Input shape of the data.
    - n_clusters (int, optional): Number of clusters for classification. Default is 10.
    - n_latent_dims (int, optional): Dimensionality of the latent space. Default is 2.
    - layer_units (list, optional): Units for each layer in the autoencoder. Default is [10, 5].
    - layer_units_classifier (list, optional): Units for each layer in the classifier. Default is [2].
    - n_pred (int, optional): Number of prediction classes if `get_pred` is True. Default is 10.
    - last_activation (str, optional): Activation for the last layer of the autoencoder. Default is "sigmoid".
    - post_loc_init_scale (float, optional): Initial scale for the posterior's location. Default is 0.1.
    - prior_scale (float, optional): Scale for the prior distribution. Default is 0.25.
    - kl_weight (float, optional): Weight for KL divergence loss. Default is 1e-5.
    - get_pred (bool, optional): Predict class labels alongside clusters. Default is False.
    - get_recon_cluster (bool, optional): Retrieve cluster prediction from reconstruction. Default is False.
    - name (str, optional): Model's name. Default is "ae".

    Attributes:
    ------------
    Various components of the model such as the encoder, decoder, and classifiers are stored as attributes.

    Methods:
    ------------
    - call(inputs, training=None): Performs a forward pass of the model.
    - compile(...): Configures the model for training.
    - train_step(data): Defines a single training step for the model.
    - test_step(data): Defines a single test (or validation) step for the model.

    Note:
    The model is designed to handle input data as a tuple of (count matrix, clusters). If enabled (via `get_pred`),
    it can also take labels for supervised training. Outputs include the reconstructed data and the predictions 
    based on latent and reconstructed representations.
    """

    def __init__(self, 
                 in_shape: tuple,
                 n_clusters: int=10,
                 n_latent_dims: int = 2, 
                 layer_units: list = [10,5], 
                 layer_units_classifier:list = [2],
                 n_pred: int = 10,
                 last_activation: str = "sigmoid",
                 post_loc_init_scale: float=0.1,
                 prior_scale: float=0.25,
                 kl_weight: float=1e-5, 
                 get_pred = False,
                 get_recon_cluster = False,
                 name='ae', 
                 **kwargs):
        super(DomainEnhancingAutoencoderClassifier, self).__init__(name=name, **kwargs)

        self.in_shape = in_shape 
        self.n_clusters = n_clusters 
        self.n_latent_dims = n_latent_dims 
        self.layer_units = layer_units
        self.last_activation = last_activation
        self.get_pred = get_pred
        self.n_pred = n_pred
        self.layer_units_classifier = layer_units_classifier
        self.get_recon_cluster = get_recon_cluster

        # RE encoder
        self.re_encoder = RandomEffectEncoder(n_latent_dims=self.n_latent_dims, 
                            layer_units=self.layer_units,
                            post_loc_init_scale=post_loc_init_scale,
                            prior_scale=prior_scale,
                            kl_weight=kl_weight)
        
        # RE decoder: weights not tied
        self.re_decoder = RandomEffectDecoder(in_shape=self.in_shape,
                                layer_units = self.layer_units,
                                last_activation = self.last_activation,
                                post_loc_init_scale=post_loc_init_scale,
                                prior_scale=prior_scale,
                                kl_weight=kl_weight)
        # The latent classifier returns class predictions in addition to cluster predictions if get_pred =True
        self.re_latent_classifier = Classifier(n_clusters=self.n_clusters,layer_units = self.layer_units_classifier,n_pred = self.n_pred, get_pred = self.get_pred)
        
        # get cluster prediction from reconstruction
        if self.get_recon_cluster:
            self.re_recon_classifier = Classifier(n_clusters=self.n_clusters,layer_units = self.layer_units_classifier, get_pred = False)
    def call(self, inputs, training=None):
        
        if len(inputs) != 2:
            raise ValueError('Model inputs need to be a tuple of (count matrix, clusters)')

        x, z = inputs

        # Encode inputs  
        latent = self.re_encoder((x, z), training=training)

        # Reconstruct image from latents
        recon = self.re_decoder((latent, z), training=training)

        output_dict = {'recon': recon}

        # Apply latent classifier 
        latent_outs = self.re_latent_classifier(latent)
        if self.get_pred:
            # The latent classifier returns class predictions in addition to cluster predictions if get_pred=True
            pred_y, pred_c_latent = latent_outs
            output_dict['pred_y'] = pred_y
            output_dict['pred_c_latent'] = pred_c_latent
        else:
            pred_c_latent = latent_outs 
            output_dict['pred_c_latent'] = pred_c_latent

        if self.get_recon_cluster:
            # Cluster predictions from reconstructed counts
            pred_c_recon = self.re_recon_classifier(recon)
            output_dict['pred_c_recon'] = pred_c_recon

        return output_dict




    def compile(self,
            loss_recon=tf.keras.losses.MeanSquaredError(),
            loss_multiclass=tf.keras.losses.CategoricalCrossentropy(),
            metric_multiclass=tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
            optimizer=tf.keras.optimizers.Adam(lr=0.0001),
            loss_recon_weight=1.0,
            loss_class_weight=0.01,
            loss_latent_cluster_weight=0.001,
            loss_recon_cluster_weight=0.001):

        super().compile()

        self.loss_recon = loss_recon
        # the loss multiclass will be used for multiclass classification (cluster, class pred, etc)
        self.loss_multiclass = loss_multiclass
        self.optimizer = optimizer
        # loss weights
        self.loss_latent_cluster_weight = loss_latent_cluster_weight
        self.loss_recon_weight = loss_recon_weight
                  

        # Loss trackers (mean loss across all the batches)
        self.loss_recon_tracker = tf.keras.metrics.Mean(name='recon_loss')
        self.loss_latent_cluster_tracker = tf.keras.metrics.Mean(name='la_clus_loss')        
        self.loss_kl_tracker = tf.keras.metrics.Mean(name='kld')
        self.loss_total_tracker = tf.keras.metrics.Mean(name='total_loss')

        if self.get_pred:
            self.metric_multiclass = metric_multiclass
            self.loss_class_weight = loss_class_weight
            self.loss_class_tracker = tf.keras.metrics.Mean(name='class_loss')

        if self.get_recon_cluster:
            self.loss_recon_cluster_weight = loss_recon_cluster_weight  
            self.loss_recon_cluster_tracker = tf.keras.metrics.Mean(name='recon_clus_loss')
    @property
    def metrics(self):
        metrics_list = [self.loss_recon_tracker,
                    self.loss_latent_cluster_tracker,
                    self.loss_kl_tracker,
                    self.loss_total_tracker]
        if self.get_pred:
            metrics_list = metrics_list +[self.loss_class_tracker,
                    self.metric_multiclass]
        elif self.get_recon_cluster: 

            metrics_list = metrics_list +[self.loss_recon_cluster_tracker]
        return metrics_list

    def _compute_update_loss(self, loss_recon, loss_latent_cluster, loss_recon_cluster=None,loss_class = None,
                             training=True):
        '''Compute total loss and update loss running means'''
        
        #update loss
        if (self.get_pred)&(loss_class is not None):
            self.loss_class_tracker.update_state(loss_class)

        if (self.get_recon_cluster)&(loss_recon_cluster is not None):
            self.loss_recon_cluster_tracker.update_state(loss_recon_cluster)

        self.loss_recon_tracker.update_state(loss_recon)
        self.loss_latent_cluster_tracker.update_state(loss_latent_cluster)
        
        
        if training:
            # The encoder and decoder have RandomEffect Layers, which inherit the properties of tpl.DenseVariational. 
            # This layer adds the kld as regularization loss to the model. The regularizations are stored in model.losses. 
            # Since there are more than one RElayers, we get the mean of all of them. 
            kld = tf.reduce_mean(self.re_encoder.losses) + tf.reduce_mean(self.re_decoder.losses)
            self.loss_kl_tracker.update_state(kld)
        else:
            # KLD can't be computed at inference time because posteriors are simplified to 
            # point estimates
            kld = 0

        loss_total = (self.loss_recon_weight*loss_recon)  + (self.loss_latent_cluster_weight * loss_latent_cluster)+kld
        if (self.get_pred)&(loss_class is not None):
            loss_total = loss_total + (self.loss_class_weight * loss_class)
        if (self.get_recon_cluster)&(loss_recon_cluster is not None):
            loss_total = loss_total + (self.loss_recon_cluster_weight * loss_recon_cluster)

        self.loss_total_tracker.update_state(loss_total)
        
        return loss_total
    def train_step(self, data):
        #missing to edit this part
        #load data
        x, clusters = data[0]

        if self.get_pred:
            _, labels = data[1]

        sample_weights = None if len(data) != 3 else data[2]


        # Train the rest of the model
        with tf.GradientTape() as gt:
            # Apply RE autoencoder: encoder + decoder
            outputs = self((x, clusters), training=True)
            recon = outputs['recon']
            pred_c_latent = outputs['pred_c_latent']

            if self.get_pred:
                pred_y = outputs['pred_y']
                # Multiclass loss
                loss_class = self.loss_multiclass(labels, pred_y)
            else:
                loss_class = None
                

            if self.get_recon_cluster:
                pred_c_recon_1 = outputs['pred_c_recon']
                loss_recon_cluster_1 = self.loss_multiclass(clusters, pred_c_recon_1)
            else:
                loss_recon_cluster_1 = None


            # mse loss
            loss_recon = self.loss_recon(x, recon)
            loss_latent_cluster = self.loss_multiclass(clusters, pred_c_latent)
            

            loss_total = self._compute_update_loss(loss_recon = loss_recon,
                                                         loss_latent_cluster =  loss_latent_cluster,
                                                         loss_recon_cluster = loss_recon_cluster_1,
                                                         loss_class = loss_class)

            
        # get trainable variables
        lsWeights = self.re_encoder.trainable_variables + self.re_decoder.trainable_variables
        # if the weight of loss_latent_cluster_weight>0, add it to the trainable variables
        if self.loss_latent_cluster_weight>0:
            lsWeights = lsWeights + self.re_latent_classifier.trainable_variables
        if self.get_recon_cluster:
            
                lsWeights = lsWeights + self.re_recon_classifier.trainable_variables

            
        # backpropagate
        grads = gt.gradient(loss_total, lsWeights)
        self.optimizer.apply_gradients(zip(grads, lsWeights))

        if self.get_pred:
            # Update metrics
            self.metric_multiclass.update_state(labels, pred_y)
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        #load data
        x, clusters = data[0]

        if self.get_pred:
            _, labels = data[1]

        sample_weights = None if len(data) != 3 else data[2]
                        
        outputs = self((x, clusters), training=False)
        recon = outputs['recon']
        pred_c_latent = outputs['pred_c_latent']

        if self.get_pred:
            pred_y = outputs['pred_y']
                # Multiclass loss
            loss_class = self.loss_multiclass(labels, pred_y)
        else:
            loss_class = None
                

        if self.get_recon_cluster:
            pred_c_recon_1 = outputs['pred_c_recon']
            loss_recon_cluster_1 = self.loss_multiclass(clusters, pred_c_recon_1)
        else:
            loss_recon_cluster_1 = None
        loss_recon = self.loss_recon(x, recon)        
        loss_latent_cluster = self.loss_multiclass(clusters, pred_c_latent)
        loss_total = self._compute_update_loss(loss_recon = loss_recon,
                                                         loss_latent_cluster =  loss_latent_cluster,
                                                         loss_recon_cluster = loss_recon_cluster_1,
                                                         loss_class = loss_class, training=False)
        if self.get_pred:
            # Update metrics
            self.metric_multiclass.update_state(labels, pred_y)
        return {m.name: m.result() for m in self.metrics}



class MixedEffectAuxClassifier(tkl.Layer):
           
    def __init__(self, 
                 units: int=2,
                 n_pred: int = 10,
                 post_loc_init_scale: float=0.1,
                 prior_scale: float=0.25,
                 kl_weight: float=1e-5,
                 add_re_2_meclass = False,
                 name='auxclassifier', **kwargs):
        """Mixed effects dense classifier with one hidden layer 
        and softmax output. Adding an re_layer is optional. Dafult: add_re_layer = False. Use in old version.
        """
      
        super().__init__(name=name, **kwargs)
        
        self.units = units
        self.n_pred = n_pred 
        self.add_re_2_meclass = add_re_2_meclass #add RE layer to Mixed effects classifier   
        self.dense_layer = tkl.Dense(self.units, activation='selu', name=name + '_dense')
        if self.add_re_2_meclass:
            self.re_layer = ClusterScaleBiasBlock(self.units, post_loc_init_scale = post_loc_init_scale,
                                                    prior_scale = prior_scale,
                                                    kl_weight = kl_weight,
                                                    name = name + '_re_layer')
            self.act = Activation('selu')
        self.dense_out = Dense(self.n_pred, activation='softmax', name=name + '_out')
        
        
    def call(self, inputs, training=None):
        x, z = inputs
        x = self.dense_layer(x)
        if self.add_re_2_meclass:
            x = self.re_layer((x, z), training=training)
            x = self.act(x)
        y = self.dense_out(x)
  
        return y

    
    def get_config(self):
        return {'units': self.units}

class MixedEffectsmodule(tf.keras.layers.Layer):
    """Concatenates Mixed effects latent space = Fixed Effects latent + Random Effects Latent. Used in old version
        Applies ME classifier on the latent space"""
    
    def __init__(self, 
                 n_latent_dims: int = 2, 
                 layer_units: list = [10,5], 
                 n_pred: int = 4,
                 post_loc_init_scale: float=0.1,
                 prior_scale: float=0.25,
                 kl_weight: float=1e-5, 
                 add_re_2_meclass = False,
                 name='me_module', 
                 **kwargs):

        super(MixedEffectsmodule, self).__init__(name=name, **kwargs)
        self.n_latent_dims = n_latent_dims 
        self.layer_units = layer_units
        self.n_pred = n_pred
        #add RE layer to ME classifier
        self.add_re_2_meclass = add_re_2_meclass

        self.concat2subnets = tf.keras.layers.Concatenate(axis=-1, name=name + 'concat_2subnets')
        # define hidden layers
        self.dense_hidden_layers = {}
        for i, n_units in enumerate(self.layer_units):
            key_name = "dense_" + str(i)
            self.dense_hidden_layers[key_name] = Dense(units=n_units, activation="selu", name=key_name)

        self.dense_me_latent = Dense(units=self.n_latent_dims, activation="selu", name="dense_me_latent")
        
        self.me_classifier = MixedEffectAuxClassifier(units = self.n_latent_dims,
                                n_pred = self.n_pred,
                                post_loc_init_scale= post_loc_init_scale,
                                prior_scale = prior_scale,
                                kl_weight = kl_weight,
                                add_re_2_meclass = self.add_re_2_meclass ,
                                name='meauxclassifier',
                                 **kwargs)
    def call(self, inputs, training=None):
        fe_latent, re_latent,z = inputs
        
        x = self.concat2subnets([fe_latent, re_latent])
        # apply hidden layers
        for key, layer in self.dense_hidden_layers.items():
            x = layer(x)
        # I will take me_latent after apply a dense layer to me_latent = fe_latent +re_latent. However, this layer may be optional
        me_latent = self.dense_me_latent(x)
        
        # me_pred_y is always done
        me_outputs = self.me_classifier((me_latent,z))
        return me_outputs   

class MixedEffectsEncoder(tf.keras.layers.Layer):
    """
    A TensorFlow Keras layer that concatenates fixed effects (FE) and random effects (RE) latent spaces,
    and optionally applies a random effects (RE) layer. The resulting mixed effects latent space is 
    then processed through a series of dense hidden layers.

    Parameters:
    - n_latent_dims (int): The number of dimensions in the mixed effects latent space.
    - layer_units (list of int): The number of units in each dense hidden layer.
    - post_loc_init_scale (float): Initial scale for the location in the post-RE layer, 
                                   used if an RE layer is added.
    - prior_scale (float): Scale of the prior in the RE layer, used if an RE layer is added.
    - kl_weight (float): Weight of the KL divergence in the loss, used if an RE layer is added.
    - add_re_2_meclass (bool): Determines whether to add an RE layer to the Mixed Effects Classifier.
    - name (str): Name of the layer.
    - **kwargs: Additional keyword arguments for the base Layer class.

    This encoder first concatenates the FE and RE latent spaces. It then processes the concatenated latent
    space through a series of dense hidden layers defined in `layer_units`. If `add_re_2_meclass` is True,
    an RE layer is applied after the dense hidden layers. The output is a mixed effects latent space that
    can be used for further processing or classification.

    The `call` method:
    Takes inputs `fe_latent`, `re_latent`, and `z`, and processes them through the encoder to produce the
    mixed effects latent space. If `add_re_2_meclass` is True, `z` is used in the RE layer.
    
    Inputs:
    - fe_latent: The latent representation of the fixed effects.
    - re_latent: The latent representation of the random effects.
    - z: Additional features or information, used if an RE layer is added.

    Returns:
    - me_latent: The resulting mixed effects latent space.
    """
    
    def __init__(self, 
                 n_latent_dims: int = 2, 
                 layer_units: list = [10,5], 
                 post_loc_init_scale: float=0.1,
                 prior_scale: float=0.25,
                 kl_weight: float=1e-5, 
                 add_re_2_meclass = False,
                 name='me_encoder', 
                 **kwargs):

        super(MixedEffectsEncoder, self).__init__(name=name, **kwargs)
        self.n_latent_dims = n_latent_dims 
        self.layer_units = layer_units
        #add RE layer to ME classifier
        self.add_re_2_meclass = add_re_2_meclass

        self.concat2subnets = tf.keras.layers.Concatenate(axis=-1, name=name + 'concat_fe_re_latent')
        # define hidden layers
        self.dense_hidden_layers = {}
        for i, n_units in enumerate(self.layer_units):
            key_name = "dense_" + str(i)
            self.dense_hidden_layers[key_name] = Dense(units=n_units, activation="selu", name=key_name)

        if self.add_re_2_meclass:
            self.re_layer = ClusterScaleBiasBlock(self.n_latent_dims, post_loc_init_scale = post_loc_init_scale,
                                                    prior_scale = prior_scale,
                                                    kl_weight = kl_weight,
                                                    name = name + '_re_layer')
            self.act = Activation('selu')

        self.dense_me_latent = Dense(units=self.n_latent_dims, activation="selu", name="dense_me_latent")
        

    def call(self, inputs, training=None):
        # fe_latent, re_latent,z = inputs
        fe_latent = inputs["fe_latent"]
        re_latent = inputs["re_latent"]
        if self.add_re_2_meclass:
            z = inputs["z"]

        
        x = self.concat2subnets([fe_latent, re_latent])
        # apply hidden layers
        for key, layer in self.dense_hidden_layers.items():
            x = layer(x)
        # Optional, add re layer
        if self.add_re_2_meclass:
            x = self.re_layer((x, z), training=training)
            x = self.act(x)

        me_latent = self.dense_me_latent(x)
        
        return me_latent  


class MixedEffectsModel(tf.keras.Model):
    """
    MixedEffectsModel. It is a mixed effects classifier which processes inputs through a Mixed Effects Encoder
    and a dense output layer for classification. It's designed to handle both fixed effects (FE)
    and random effects (RE) latent spaces, making it suitable for scenarios where
    both fixed and random effects are considered.

    Parameters:
    - n_latent_dims (int): The number of dimensions in the mixed effects latent space
                           created by the Mixed Effects Encoder.
    - layer_units (list of int): The number of units in each dense hidden layer within
                                 the Mixed Effects Encoder.
    - n_pred (int): The number of units in the final dense output layer, typically corresponding
                    to the number of classes for classification.
    - post_loc_init_scale (float): Initial scale for the location in the post-RE layer within
                                   the Mixed Effects Encoder, used if an RE layer is added.
    - prior_scale (float): Scale of the prior in the RE layer within the Mixed Effects Encoder,
                           used if an RE layer is added.
    - kl_weight (float): Weight of the KL divergence in the loss within the Mixed Effects Encoder,
                         used if an RE layer is added.
    - add_re_2_meclass (bool): Determines whether to add an RE layer to the Mixed Effects Classifier
                               within the Mixed Effects Encoder.
    - name (str): Name of the model.
    - **kwargs: Additional keyword arguments for the base Model class.

    The model encapsulates a Mixed Effects Encoder for processing the FE and RE latent spaces,
    followed by a dense output layer with softmax activation for classification.

    The `call` method:
    Processes the inputs through the Mixed Effects Encoder and then through the dense output layer.
    
    Inputs:
    - fe_latent: The latent representation of the fixed effects.
    - re_latent: The latent representation of the random effects.
    - z: Additional features or information, used in the RE layer of the Mixed Effects Encoder
         if `add_re_2_meclass` is True.

    Returns:
    - y: The classification output, with probabilities for each class.
    """
    def __init__(self, 
                 n_latent_dims: int = 2, 
                 layer_units: list = [10,5], 
                 n_pred: int = 10,
                 post_loc_init_scale: float=0.1,
                 prior_scale: float=0.25,
                 kl_weight: float=1e-5, 
                 add_re_2_meclass = False,
                 name='mec', 
                 **kwargs):
        super(MixedEffectsModel, self).__init__(**kwargs)
        
        self.n_latent_dims = n_latent_dims 
        self.layer_units = layer_units
        self.n_pred = n_pred

        
        # MixedEffectsmodule
        self.encoder =  MixedEffectsEncoder(n_latent_dims = self.n_latent_dims,
                            layer_units = self.layer_units,
                            post_loc_init_scale = post_loc_init_scale,
                            prior_scale = prior_scale,
                            kl_weight = kl_weight,
                            add_re_2_meclass = add_re_2_meclass,
                            name = 'me_encoder', 
                            **kwargs)

        self.dense_out = Dense(self.n_pred, activation='softmax', name=name + '_out')
    def call(self, inputs,training=None):

        # fe_latent, re_latent,z = inputs
        me_latent = self.encoder(inputs,training=training)
        # dense out with softmax activation
        y = self.dense_out(me_latent)
        return y



class MixedEffectsModel_old(tf.keras.Model):
    """model that receives FE latent and RE latent as inputs. Returns celltype classification. Old version"""
    def __init__(self, 
                 n_latent_dims: int = 2, 
                 layer_units: list = [10,5], 
                 n_pred: int = 4,
                 post_loc_init_scale: float=0.1,
                 prior_scale: float=0.25,
                 kl_weight: float=1e-5, 
                 add_re_2_meclass = False,
                 name='me_module', 
                 **kwargs):
        super(MixedEffectsModel_old, self).__init__(**kwargs)
        
        self.n_latent_dims = n_latent_dims 
        self.layer_units = layer_units
        self.n_pred = n_pred
        #add RE layer to ME classifier
        self.add_re_2_meclass = add_re_2_meclass
        
        # MixedEffectsmodule
        self.me =  MixedEffectsmodule(n_latent_dims = self.n_latent_dims,
                            layer_units = self.layer_units,
                            n_pred = self.n_pred,
                            post_loc_init_scale = post_loc_init_scale,
                            prior_scale = prior_scale,
                            kl_weight = kl_weight,
                            add_re_2_meclass = self.add_re_2_meclass,
                            name = 'me_module', 
                            **kwargs)
    def call(self, inputs):
        fe_latent, re_latent,z = inputs
        y = self.me((fe_latent, re_latent,z))
        return y




class MixedEffectsAEC(tf.keras.Model):
    "Big model. Too many losses"
    
    def __init__(self, 
                 in_shape: tuple,
                 n_clusters: int=10,
                 n_latent_dims: int = 2, 
                 layer_units: list = [10,5], 
                 layer_units_classifier:list = [2],
                 n_pred: int = 4,
                 last_activation: str = "sigmoid",
                 post_loc_init_scale: float=0.1,
                 prior_scale: float=0.25,
                 kl_weight: float=1e-5, 
                 get_pred = False,
                 add_re_2_meclass = False,
                 name='ae', 
                 **kwargs):

        super(MixedEffectsAEC, self).__init__(name=name, **kwargs)
        self.in_shape = in_shape 
        self.n_clusters = n_clusters 
        self.n_latent_dims = n_latent_dims 
        self.layer_units = layer_units
        self.last_activation = last_activation
        self.get_pred = get_pred
        self.n_pred = n_pred
        self.layer_units_classifier = layer_units_classifier
        #add RE layer to ME classifier
        self.add_re_2_meclass = add_re_2_meclass        

        # Initialize the DomainAdversarialAE
        self.ae_fe = DomainAdversarialAE(in_shape=self.in_shape,
                         n_clusters=self.n_clusters,
                         n_latent_dims=self.n_latent_dims,
                         layer_units=self.layer_units,
                         last_activation=self.last_activation,
                         n_pred=self.n_pred,
                         layer_units_latent_classifier=self.layer_units_classifier,
                         get_pred=self.get_pred,
                         name="ae_fe",
                         **kwargs)
        
        # Initialize the DomainEnhancingAutoencoderClassifier (for random effects)
        self.ae_re = DomainEnhancingAutoencoderClassifier(in_shape=self.in_shape,
                                            n_clusters=self.n_clusters,
                                            n_latent_dims=self.n_latent_dims,
                                            layer_units=self.layer_units,
                                            layer_units_classifier=self.layer_units_classifier,
                                            n_pred=self.n_pred,
                                            last_activation=self.last_activation,
                                            post_loc_init_scale=post_loc_init_scale,
                                            prior_scale=prior_scale,
                                            kl_weight=kl_weight,
                                            get_pred=self.get_pred,
                                            name="ae-me",
                                            **kwargs)

        self.me =  MixedEffectsmodule(n_latent_dims = self.n_latent_dims,
                            layer_units = self.layer_units,
                            n_pred = self.n_pred,
                            post_loc_init_scale = post_loc_init_scale,
                            prior_scale = prior_scale,
                            kl_weight = kl_weight,
                            add_re_2_meclass = self.add_re_2_meclass,
                            name = 'me_module', 
                            **kwargs)

    def call(self, inputs, training=None):
        x, z = inputs
        ###################################################### DomainEnhancingAutoencoderClassifier (for RE subnetwork)
        # encode inputs  
        re_latent = self.ae_re.re_encoder((x, z), training=training)

        # Reconstruct image from latents
        re_recon = self.ae_re.re_decoder((re_latent, z), training=training)

        # cluster predictions from reconstructed counts: vector of n samples * n clusters with the probability of each sample being of each cluster
        re_pred_c_recon = self.ae_re.re_recon_classifier(re_recon)   


        ####################################################### Domain Adversarial for fixed effects
        # encoder
        encoder_activations = self.ae_fe.encoder(x)
        # apply adversary to encoder activations (decoder shares weights with encoder)
        fe_pred_cluster = self.ae_fe.adversary(encoder_activations)
        # latent space is the last activation layer
        fe_latent = encoder_activations[-1]
        # decoder is applied to latent
        fe_recon = self.ae_fe.decoder(fe_latent)

        ##################################################### Mixed effects classifier
        # me_pred_y is always done
        me_outputs = self.me((fe_latent, re_latent))    

        ########### Add classification loss if you need classification predictions
        if self.get_pred:

            # The latent classifiers return class predictions in addition to cluster predictions if get_pred =True
            # For re subnet (2 outputs: one for class and another one for cluster pred (z))
            re_pred_y, re_pred_c_latent = self.ae_re.re_latent_classifier(re_latent)
            # For fe subnet
            fe_pred_class = self.ae_fe.latent_classifier(fe_latent)
            
            # outputs
            # re outputs
            re_outputs = (re_recon, re_pred_y, re_pred_c_latent, re_pred_c_recon)
            # fe outputs
            fe_outputs = (fe_recon, fe_pred_class, fe_pred_cluster)
           
            
            return (fe_outputs,re_outputs,me_outputs) 
        else: 
            #For re subnet: (1 output for cluster pred (z))
            re_pred_c_latent = self.ae_re.re_latent_classifier(re_latent)
            # outputs
            # re outputs
            re_outputs = (re_recon, re_pred_c_latent, re_pred_c_recon )
            # fe outputs
            fe_outputs = (fe_recon, fe_pred_cluster)
       
            return (fe_outputs,re_outputs,me_outputs) 


    def compile(self,
                loss_recon=tf.keras.losses.MeanSquaredError(),
                loss_multiclass=tf.keras.losses.CategoricalCrossentropy(),
                metric_multiclass=tf.keras.metrics.CategoricalAccuracy(name='acc'),
                opt_ae_fe=tf.keras.optimizers.Adam(lr=0.0001),
                opt_adversary=tf.keras.optimizers.Adam(lr=0.0001),
                opt_me=tf.keras.optimizers.Adam(lr=0.0001),
                loss_recon_weight_fe=1.0,
                loss_gen_weight_fe=0.05,
                loss_class_weight_fe=0.01, 
                loss_recon_weight_re=1.0,                           
                loss_class_weight_re=0.01,
                loss_latent_cluster_weight_re=0.001,
                loss_recon_cluster_weight_re=0.001):

        super().compile()
        ################################# Define common attributes
        # loss class can be used for fe, re and me
        self.loss_multiclass = loss_multiclass
        ################################ Define fe compile attributes

        # define fe optimizers
        self.opt_ae_fe = opt_ae_fe
        self.opt_adversary = opt_adversary
        
        # Define fe losses (for the others I will use loss multiclass)
        self.loss_recon_fe = loss_recon       
        self.loss_adv = loss_multiclass  
    
        # define loss weights
        self.loss_recon_weight_fe = loss_recon_weight_fe
        self.loss_gen_weight_fe = loss_gen_weight_fe       

        # track mean loss
        self.loss_recon_tracker_fe = tf.keras.metrics.Mean(name='recon_loss_fe')
        self.loss_adv_tracker_fe = tf.keras.metrics.Mean(name='adv_loss_fe')
        self.loss_total_tracker_fe = tf.keras.metrics.Mean(name='total_loss_fe')

        # define metrics
        self.metric_adv = metric_multiclass


        if self.get_pred: # define latent class loss, metric and weights
            self.metric_multiclass_fe = metric_multiclass
            self.loss_class_weight_fe = loss_class_weight_fe
            self.loss_class_tracker_fe = tf.keras.metrics.Mean(name='class_loss_fe')
        
        ################################ Define me + re compile attributes
        
        # define me optimizer
        self.opt_me = opt_me

        # Define losses (only recon, for others I will use loss multiclass)
        self.loss_recon_re = loss_recon 
        # Define losses weights  
        self.loss_recon_weight_re = loss_recon_weight_re 
        self.loss_latent_cluster_weight_re = loss_latent_cluster_weight_re
        self.loss_recon_cluster_weight_re =  loss_recon_cluster_weight_re


        # Loss trackers re (mean loss across all the batches)
        self.loss_recon_tracker_re = tf.keras.metrics.Mean(name='recon_loss_re')
        self.loss_latent_cluster_tracker_re = tf.keras.metrics.Mean(name='la_clus_loss_re')
        self.loss_recon_cluster_tracker_re = tf.keras.metrics.Mean(name='recon_clus_loss_re')
        self.loss_kl_tracker_re = tf.keras.metrics.Mean(name='kld')
        self.loss_total_tracker_re = tf.keras.metrics.Mean(name='total_loss_re')
        
        # Loss tracker me
        self.loss_class_tracker_me = tf.keras.metrics.Mean(name='class_loss_me')
        self.loss_total_tracker_me = tf.keras.metrics.Mean(name='total_loss_me')

        # define metric_muticlass_me
        self.metric_multiclass_me = metric_multiclass

        if self.get_pred:
            #self.metric_multiclass_re = metric_multiclass
            self.loss_class_weight_re = loss_class_weight_re
            self.loss_class_tracker_re = tf.keras.metrics.Mean(name='class_loss_re')

    @property
    def metrics(self):
        if self.get_pred:
            return [self.loss_recon_tracker_fe,
                self.loss_class_tracker_fe,
                self.loss_adv_tracker_fe,
                self.loss_total_tracker_fe,
                self.loss_recon_tracker_re,
                self.loss_latent_cluster_tracker_re,
                self.loss_recon_cluster_tracker_re,
                self.loss_kl_tracker_re,
                self.loss_class_tracker_re,
                self.loss_total_tracker_re,
                self.loss_class_tracker_me,
                self.loss_total_tracker_me,
                self.metric_adv,
                self.metric_multiclass_fe,
                self.metric_multiclass_me]
        else:
            return [self.loss_recon_tracker_fe,
                self.loss_adv_tracker_fe,
                self.loss_total_tracker_fe,
                self.loss_recon_tracker_re,
                self.loss_latent_cluster_tracker_re,
                self.loss_recon_cluster_tracker_re,
                self.loss_kl_tracker_re,
                self.loss_total_tracker_re,
                self.loss_class_tracker_me,
                self.loss_total_tracker_me,
                self.metric_adv,
                self.metric_multiclass_fe,
                self.metric_multiclass_me]
    def _compute_update_loss_re(self, loss_recon_re, loss_latent_cluster_re, loss_recon_cluster_re,loss_class_re = None,
                             training=True):
        '''Compute total loss and update loss running means. For random effects subnet'''
        
        #update loss
        if (self.get_pred)&(loss_class_re is not None):
            self.loss_class_tracker_re.update_state(loss_class_re)

        self.loss_recon_tracker_re.update_state(loss_recon_re)
        self.loss_latent_cluster_tracker_re.update_state(loss_latent_cluster_re)
        self.loss_recon_cluster_tracker_re.update_state(loss_recon_cluster_re)
        
        if training:
            # The encoder and decoder have RandomEffect Layers, which inherit the properties of tpl.DenseVariational. 
            # This layer adds the kld as regularization loss to the model. The regularizations are stored in model.losses. 
            # Since there are more than one RElayers, we get the mean of all of them. 
            kld = tf.reduce_mean(self.ae_re.re_encoder.losses) + tf.reduce_mean(self.ae_re.re_decoder.losses)
            self.loss_kl_tracker_re.update_state(kld)
        else:
            # KLD can't be computed at inference time because posteriors are simplified to 
            # point estimates
            kld = 0
        if (self.get_pred)&(loss_class_re is not None):
            loss_total_re = loss_recon_re \
                + (self.loss_class_tracker_re * loss_class_re) \
                + (self.loss_latent_cluster_weight_re * loss_latent_cluster_re) \
                + (self.loss_recon_cluster_weight_re * loss_recon_cluster_re) \
                + kld
        else:
            loss_total_re = loss_recon_re \
                + (self.loss_latent_cluster_weight_re * loss_latent_cluster_re) \
                + (self.loss_recon_cluster_weight_re * loss_recon_cluster_re) \
                + kld
        self.loss_total_tracker_re.update_state(loss_total_re)
        
        return loss_total_re

    def _compute_update_loss_fe(self, loss_recon_fe,loss_adv,loss_class_fe = None,training=True):
        '''Compute total loss and update loss running means. For fixed effects subnet'''
        
        #update loss
        if (self.get_pred)&(loss_class_fe is not None):
            self.loss_class_tracker_fe.update_state(loss_class_fe)

        self.loss_recon_tracker_fe.update_state(loss_recon_fe)


        if self.get_pred:
            total_loss_fe = (self.loss_recon_weight_fe * loss_recon_fe) \
                    + (self.loss_class_weight_fe * loss_class_fe) \
                    - (self.loss_gen_weight_fe * loss_adv) 
            self.loss_class_tracker_fe.update_state(loss_class_fe)
            self.loss_recon_tracker_fe.update_state(loss_recon_fe)
            self.loss_total_tracker_fe.update_state(total_loss_fe)
            return total_loss_fe 
        else:
            #compute total ae loss: (recon) - adv loss (gen)
            total_loss_fe = (self.loss_recon_weight_fe * loss_recon_fe)- (self.loss_gen_weight_fe * loss_adv)
            self.loss_recon_tracker_fe.update_state(loss_recon_fe)
            self.loss_total_tracker_fe.update_state(total_loss_fe)
            return total_loss_fe 

    def train_step(self, data):

        #load data
        x, clusters = data[0]

        if self.get_pred:
            _, labels = data[1]
        # else:
        #     labels = None

        sample_weights = None if len(data) != 3 else data[2]

        #CHECK IF THE SHAPES ARE CORRECT
        assert x.shape[0] == clusters.shape[0], "Mismatch between x and clusters"
        if self.get_pred:
            assert x.shape[0] == labels.shape[0], "Mismatch between x and labels"

        #Train adversary
        encoder_outs = self.ae_fe.encoder(x, training=True)

        #calculate adv loss
        with tf.GradientTape() as gt1:
            pred_cluster = self.ae_fe.adversary(encoder_outs)
            loss_adv = self.loss_adv(clusters, pred_cluster, sample_weight=sample_weights)
        
        #apply gradients
        grads_adv = gt1.gradient(loss_adv, self.ae_fe.adversary.trainable_variables)
        self.opt_ae_fe.apply_gradients(zip(grads_adv, self.ae_fe.adversary.trainable_variables))
        
        # Update adversarial loss tracker
        self.metric_adv.update_state(clusters, pred_cluster)
        self.loss_adv_tracker_fe.update_state(loss_adv)

        # Train autoencoder fe
        with tf.GradientTape(persistent=True) as gt2:
            #apply model   
            outputs = self.ae_fe(inputs=(x, clusters), training=True)
            if self.get_pred:
                pred_recon, pred_class, pred_cluster = outputs #(+ pred class)
            else:
                pred_recon, pred_cluster = outputs
            
            #compute individual losses
            loss_recon_fe = self.loss_recon_fe(x, pred_recon, sample_weight=sample_weights)
            loss_adv = self.loss_adv(clusters, pred_cluster, sample_weight=sample_weights)
            if self.get_pred:
                loss_class_fe = self.loss_multiclass(labels, pred_class, sample_weight=sample_weights)
                loss_total_fe = self._compute_update_loss_fe(loss_recon_fe = loss_recon_fe,loss_adv = loss_adv,loss_class_fe = loss_class_fe ,training=True)
            else:

                loss_total_fe = self._compute_update_loss_fe(loss_recon_fe = loss_recon_fe,loss_adv = loss_adv,training=True)

        if self.get_pred: # +latent classifier trainable vars
                lsWeights = self.ae_fe.encoder.trainable_variables + self.ae_fe.decoder.trainable_variables \
                + self.ae_fe.latent_classifier.trainable_variables
        else:
            lsWeights = self.ae_fe.encoder.trainable_variables + self.ae_fe.decoder.trainable_variables
        
        #backpropagate
        grads_aec = gt2.gradient(loss_total_fe, lsWeights)
        self.opt_ae_fe.apply_gradients(zip(grads_aec, lsWeights))

        # Update loss trackers
        if self.get_pred:
            self.metric_multiclass_fe.update_state(labels, pred_class)


        ###########################################Train RE + ME

        with tf.GradientTape() as gt3:
            # Apply full model (we are only going to train re_outputs and me_outputs)
            fe_outputs, re_outputs, me_outputs = self((x, clusters), training=True)


            if self.get_pred:
                recon_re, pred_y_re, pred_c_latent_re, pred_c_recon_re = re_outputs
                # RE class loss
                loss_class_re = self.loss_multiclass(labels, pred_y_re)
            else:
                recon_re, pred_c_latent_re, pred_c_recon_re = re_outputs
            # RE mse loss
            loss_recon_re = self.loss_recon_re(x, recon_re)
            # RE cluster losses
            loss_latent_cluster_re = self.loss_multiclass(clusters, pred_c_latent_re)
            loss_recon_cluster_re = self.loss_multiclass(clusters, pred_c_recon_re)

            # if self.get_pred = True: make it supervised because you take into account the y_pred, else make it unsupervised (do not take into account loss class)
            if self.get_pred:
                
                loss_total_re = self._compute_update_loss_re(loss_recon = loss_recon_re,
                                                         loss_latent_cluster =  loss_latent_cluster_re,
                                                         loss_recon_cluster = loss_recon_cluster_re,
                                                         loss_class = loss_class_re)
            else:
                loss_total_re = self._compute_update_loss_re(loss_recon = loss_recon_re,
                                                         loss_latent_cluster =  loss_latent_cluster_re,
                                                         loss_recon_cluster = loss_recon_cluster_re)
            # ME class loss
            loss_class_me = self.loss_multiclass(labels, me_outputs)
            # Total loss = ME loss + RE loss
            loss_total =   loss_class_me +loss_total_re          
        # trainable vars from RE and ME
        lsWeights = self.ae_re.re_encoder.trainable_variables + self.ae_re.re_decoder.trainable_variables \
            + self.ae_re.re_latent_classifier.trainable_variables + self.ae_re.re_recon_classifier.trainable_variables+ self.me.trainable_variables
            
        # backpropagate
        grads_re_me= gt3.gradient(loss_total, lsWeights)
        self.opt_me.apply_gradients(zip(grads_re_me, lsWeights))

        #update me losses
        self.loss_class_tracker_me.update_state(loss_class_me)
        self.loss_total_tracker_me.update_state(loss_total)

        if self.get_pred:
            # Update metrics
            self.metric_multiclass_me.update_state(labels, me_outputs)


        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Load data
        x, clusters = data[0]
        
        if self.get_pred:
            _, labels = data[1]

        # Check if the shapes are correct
        assert x.shape[0] == clusters.shape[0], "Mismatch between x and clusters"
        if self.get_pred:
            assert x.shape[0] == labels.shape[0], "Mismatch between x and labels"

        # Fixed Effects subnet
        fe_outputs = self.ae_fe(inputs=(x, clusters), training=False)
        if self.get_pred:
            pred_recon_fe, pred_class_fe, pred_cluster_fe = fe_outputs
            loss_class_fe = self.loss_multiclass(labels, pred_class_fe)
        else:
            pred_recon_fe, pred_cluster_fe = fe_outputs

        loss_recon_fe = self.loss_recon_fe(x, pred_recon_fe)
        loss_adv_fe = self.loss_adv(clusters, pred_cluster_fe)
        
        if self.get_pred:
            total_loss_fe = self._compute_update_loss_fe(loss_recon_fe, loss_adv_fe, loss_class_fe, training=False)
        else:
            total_loss_fe = self._compute_update_loss_fe(loss_recon_fe, loss_adv_fe, training=False)

        # Update metrics and trackers
        self.loss_adv_tracker_fe.update_state(loss_adv_fe)
        #self.loss_total_tracker_fe.update_state(total_loss_fe) # This is updated inside ._compute_update_loss_fe

        if self.get_pred:
            self.metric_multiclass_fe.update_state(labels, pred_class_fe)
        
        # Random Effects + Mixed Effects subnet
        _, re_outputs, me_outputs = self((x, clusters), training=False)
        
        if self.get_pred:
            recon_re, pred_y_re, pred_c_latent_re, pred_c_recon_re = re_outputs
            loss_class_re = self.loss_multiclass(labels, pred_y_re)
        else:
            recon_re, pred_c_latent_re, pred_c_recon_re = re_outputs
        
        loss_recon_re = self.loss_recon_re(x, recon_re)
        loss_latent_cluster_re = self.loss_multiclass(clusters, pred_c_latent_re)
        loss_recon_cluster_re = self.loss_multiclass(clusters, pred_c_recon_re)

        if self.get_pred:
            loss_total_re = self._compute_update_loss_re(loss_recon_re, loss_latent_cluster_re, loss_recon_cluster_re, loss_class_re, training=False)
        else:
            loss_total_re = self._compute_update_loss_re(loss_recon_re, loss_latent_cluster_re, loss_recon_cluster_re, training=False)

        loss_class_me = self.loss_multiclass(labels, me_outputs)
        #total loss me
        loss_total = loss_class_me + loss_total_re 
        # Update trackers and metrics
        self.loss_class_tracker_me.update_state(loss_class_me)
        #self.loss_total_tracker_re.update_state(loss_total_re) . This was updated inside ._compute_update_loss_re
        self.metric_multiclass_me.update_state(labels, me_outputs)
        #update me loss
        self.loss_total_tracker_me.update_state(loss_total)

        return {m.name: m.result() for m in self.metrics}





