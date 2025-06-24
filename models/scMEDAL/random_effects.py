'''
Core random effects Bayesian layers. Code adapted from Nguyen et al 2023: tinyurl.com/ARMEDCode
'''
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tkl
from tensorflow_probability import layers as tpl
from tensorflow_probability import distributions as tpd

from tensorflow_addons.layers import InstanceNormalization

def make_posterior_fn(post_loc_init_scale, post_scale_init_min, post_scale_init_range):
    def _re_posterior_fn(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        # There are n variables containing the mean of each weight and n variables
        # containing the shared s.d. for all weights
        initializer = tpl.BlockwiseInitializer([tf.keras.initializers.RandomNormal(mean=0, 
                                                                                   stddev=post_loc_init_scale), 
                                                tf.keras.initializers.RandomUniform(minval=post_scale_init_min, 
                                                                                    maxval=post_scale_init_min \
                                                                                        + post_scale_init_range),
                                                ],
                                            sizes=[n, n])

        return tf.keras.Sequential([tpl.VariableLayer(n + n, dtype=dtype, initializer=initializer),
                                    tpl.DistributionLambda(lambda t: tpd.Independent(
                                    tpd.Normal(loc=t[..., :n], scale=1e-5 + tf.nn.softplus(t[..., n:])),
                                    reinterpreted_batch_ndims=1))
                                ])
    return _re_posterior_fn


def make_fixed_prior_fn(prior_scale):
    def _prior_fn(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([tpl.DistributionLambda(lambda t: 
                                        tpd.Independent(
                                            tpd.Normal(loc=tf.zeros(n), scale=prior_scale),
                                            reinterpreted_batch_ndims=1))
                                    ])
    return _prior_fn

def make_trainable_prior_fn(prior_scale):
    def _prior_fn(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        initializer = tf.initializers.Constant(prior_scale)
        return tf.keras.Sequential([tpl.VariableLayer(n, dtype=dtype, initializer=initializer),
                                    tpl.DistributionLambda(lambda t: 
                                        tpd.Normal(loc=tf.zeros(n), scale=1e-5 + tf.nn.softplus(t)))])
    return _prior_fn

class RandomEffects(tpl.DenseVariational):
    def __init__(self, 
                 units: int=1, 
                 post_loc_init_scale: float=0.05, 
                 post_scale_init_min: float=0.05,
                 post_scale_init_range: float=0.05,
                 prior_scale: float=0.05,
                 kl_weight: float=0.001,
                 l1_weight: float=None,
                 name=None) -> None:
        """Core random effects layer, which learns cluster-specific parameters
        regularized to a zero-mean normal distribution. It takes as input a 
        one-hot encoded matrix Z indicating the cluster membership of each sample, 
        then returns a vector of cluster-specific parameters u(Z). Each parameter
        is regularized to follow zero-mean normal distribution.

        Args:
            units (int, optional): Number of parameters. Defaults to 1.
            post_loc_init_scale (float, optional): S.d. for initializing
                posterior means with a random normal distribution. Defaults to 0.05.
            post_scale_init_min (float, optional): Range lower bound for
                initializing posterior variances with a random uniform distribution.
                Defaults to 0.05.
            post_scale_init_range (float, optional): Range width for
                initializing posterior variances with a random uniform distribution. 
                Defaults to 0.05.
            prior_scale (float, optional): S.d. of prior distribution. Defaults to 0.05.
            kl_weight (float, optional): KL divergence weight. Defaults to 0.001.
            l1_weight (float, optional): L1 regularization weight. Defaults to None.
            name (str, optional): Layer name. Defaults to None.
        """        
        
        self.kl_weight = kl_weight
        self.l1_weight = l1_weight
        
        # The posterior scale is saved as a softplus transformed weight, so we
        # need to convert the given initalization args using the inverse
        # softplus
        fPostScaleMin = np.log(np.exp(post_scale_init_min) - 1)
        fPostScaleRange = np.log(np.exp(post_scale_init_range) - 1)
        
        posterior = make_posterior_fn(post_loc_init_scale, fPostScaleMin, fPostScaleRange)
        prior = make_fixed_prior_fn(prior_scale)
        
        super().__init__(units, posterior, prior, use_bias=False,
                         kl_weight=kl_weight,
                         name=name)
        
    def call(self, inputs, training=None):
        
        if training == False:
            # In testing mode, use the posterior means 
            if self._posterior.built == False:
                self._posterior.build(inputs.shape)
            if self._prior.built == False:
                self._prior.build(inputs.shape)
            
            # First half of weights contains the posterior means
            nWeights = self.weights[0].shape[0]
            w = self.weights[0][:(nWeights // 2)]
                        
            prev_units = self.input_spec.axes[-1]

            kernel = tf.reshape(w, shape=tf.concat([
                tf.shape(w)[:-1],
                [prev_units, self.units],
            ], axis=0))
            #print("\nrandom effects inputs", inputs, "dtype:", inputs.dtype)
            #print("\nrandom effects kernel", kernel, "dtype:", kernel.dtype)
            inputs = tf.cast(inputs, tf.float32)

            outputs = tf.matmul(inputs, kernel)

            if self.activation is not None:
                outputs = self.activation(outputs)  # pylint: disable=not-callable
        else:
            outputs = super().call(inputs)
        
        if self.l1_weight:
            # First half of weights contains the posterior means
            nWeights = self.weights[0].shape[0]
            postmeans = self.weights[0][:(nWeights // 2)]
            
            self.add_loss(self.l1_weight * tf.reduce_sum(tf.abs(postmeans)))
        
        return outputs

       

class ClusterScaleBiasBlock(tf.keras.Model):
    
    def __init__(self,
                 n_features, 
                 post_loc_init_scale=0.25,
                 prior_scale=0.25,
                 kl_weight=0.001,
                 name='cluster', 
                 **kwargs):
        """Layer applying cluster-specific random scales and biases to the
        output of a convolution layer.
        
        This layer learns cluster-specific scale vectors 'gamma(Z)' and bias
        vectors 'beta(Z)', where Z is the one-hot. These vectors have length 
        equal to the number of filters in the preceding convolution layer. 
        After instance-normalzing the input x, the following operation is 
        applied:
            
            (1 + gamma) * x + beta
            
        Any activation function should be placed after this layer. Other 
        normalization layers should not be used. 

        Args:
            n_features (int): Number of filters in preceding convolution layer.
            post_loc_init_scale (float, optional): S.d. for initializing
                posterior means with a random normal distribution. Defaults to 0.25.
            prior_scale (float, optional): S.d. of normal prior distribution. Defaults to 0.25.
            gamma_dist (bool, optional): Use a gamma prior distribution (not
                fully tested). Defaults to False.
            kl_weight (float, optional): KL divergence weight. Defaults to 0.001.
            name (str, optional): Layer name. Defaults to 'cluster'.
        """        
        super(ClusterScaleBiasBlock, self).__init__(name=name, **kwargs)
        
        self.n_features = n_features
        self.post_loc_init_scale = post_loc_init_scale
        self.prior_scale = prior_scale

        
        self.kl_weight = kl_weight
        
        # self.instance_norm = InstanceNormalization(center=True, 
                                                #    scale=True, 
                                                #    name=name + '_instance_norm')
        self.bn = tf.keras.layers.BatchNormalization(name = name + '_batch_norm')


        self.gammas = RandomEffects(n_features, 
                                        post_loc_init_scale=post_loc_init_scale,
                                        post_scale_init_min=0.01, 
                                        post_scale_init_range=0.005, 
                                        prior_scale=prior_scale, 
                                        kl_weight=kl_weight,
                                        name=name + '_gammas')
        self.betas = RandomEffects(n_features, 
                                    post_loc_init_scale=post_loc_init_scale,
                                    post_scale_init_min=0.01, 
                                    post_scale_init_range=0.005, 
                                    prior_scale=prior_scale, 
                                    kl_weight=kl_weight,
                                    name=name + '_betas')

    def call(self, inputs, training=None):
        x, z = inputs
        # x = self.instance_norm(x)
        # batch normalization. There was a bug when using instance_norm
        x = self.bn(x,training=training)
        g = self.gammas(z, training=training)
        b = self.betas(z, training=training)    
        # Ensure shape is batch_size x 1 x 1 x n_features
        if len(tf.shape(x)) > 2:
            new_dims = len(tf.shape(x)) - 2
            g = tf.reshape(g, [-1] + [1] * new_dims + [self.n_features])
            b = tf.reshape(b, [-1] + [1] * new_dims + [self.n_features])
        
        m = x * (1 + g)
        s = m + b
        return s
    
    def get_config(self):
        return {'post_loc_init_scale': self.post_loc_init_scale,
                'prior_scale': self.prior_scale,
                'gamma_dist': self.gamma_dist,
                'kl_weight': self.kl_weight}       