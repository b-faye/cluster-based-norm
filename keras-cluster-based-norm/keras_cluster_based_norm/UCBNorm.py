# C'est la bonne version j'ai aussi modifié sur le git
from keras import backend as K
from keras.layers import Layer, Input
import numpy as np

@utils.register_keras_serializable()
class UCBNorm(Layer):
    def __init__(self, num_components, epsilon=1e-3, momentum=0.9, **kwargs):
        self.num_components = num_components
        self.epsilon = epsilon
        self.momentum = momentum
        super(UCBNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.mean = self.add_weight(
            name='mean',
            shape=(self.num_components, self.input_dim),
            initializer=self.custom_mean_initializer,
            trainable=True
        )
        self.variance = self.add_weight(
            name='variance',
            shape=(self.num_components, self.input_dim),
            initializer=self.custom_variance_initializer,
            trainable=True
        )
        self.prior = self.add_weight(
            name='prior',
            shape=(self.num_components, 1),
            initializer=self.custom_prior_initializer,
            trainable=True
        )
        super(UCBNorm, self).build(input_shape)

    def custom_mean_initializer(self, shape, dtype=None):
        min_value = -1.0  # Minimum value for means
        max_value = 1.0  # Maximum value for means
        return K.random_uniform(shape=shape, minval=min_value, maxval=max_value, dtype=dtype)

    def custom_variance_initializer(self, shape, dtype=None):
        min_value = 0.001  # Minimum value for variance
        max_value = 0.01  # Maximum value for variance
        return K.random_uniform(shape=shape, minval=min_value, maxval=max_value, dtype=dtype)

    def custom_prior_initializer(self, shape, dtype=None):
        initial_values = K.random_uniform(shape, minval=0.01, maxval=0.99, dtype=dtype)
        normalized_values = initial_values / K.sum(initial_values, axis=0)
        return normalized_values

    def get_config(self):
        config = super(UCBNorm, self).get_config().copy()
        config.update(
            {
                "num_components": self.num_components,
                "epsilon": self.epsilon,
                "momentum": self.momentum
            })
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()
        x = inputs
        normalized_x = K.zeros_like(x)
        mean = self.mean
        var = self.variance
        prior = self.prior
        num_expand_dims = len(K.int_shape(x)) - 2

        for _ in range(num_expand_dims):
            mean = K.expand_dims(mean, axis=1)
            var = K.expand_dims(var, axis=1)

        var = K.softplus(var)
        prior = K.softmax(prior)

        for k in range(self.num_components):
            mean_k = mean[k, :]
            var_k = var[k, :]
            prior_k = prior[k, :]

            p_x_given_k = prior_k * K.exp(-0.5 * ((x - mean_k) * (1 / (var_k + self.epsilon)) * (x - mean_k)))

            p_x_given_i = K.zeros_like(p_x_given_k)
            for i in range(self.num_components):
                mean_i = mean[i, :]
                var_i = var[i, :]

                posterior_proba = prior[i] * K.exp(
                    -0.5 * ((x - mean_i) * (1 / (var_i + self.epsilon)) * (x - mean_i)))

                p_x_given_i += posterior_proba

            tau_k = p_x_given_k / (p_x_given_i + self.epsilon)
            
            if training:
                sum_tau_k = K.sum(tau_k, axis=list(range(K.ndim(x)))[:-1])
                hat_tau_k = tau_k / (sum_tau_k + self.epsilon)
    
                prod = hat_tau_k * x
                expectation = K.mean(prod, axis=list(range(K.ndim(x)))[1:-1])
                shape1 = K.int_shape(x)
                shape2 = K.int_shape(expectation)
                num_extra_dims = len(shape1) - len(shape2)
                for _ in range(num_extra_dims):
                    expectation = K.expand_dims(expectation, axis=1)
                v_i_k = x - expectation
    
                prod_bis = hat_tau_k * (prod * prod)
                variance = K.mean(prod_bis, axis=list(range(K.ndim(x)))[1:-1])
                shape1 = K.int_shape(v_i_k)
                shape2 = K.int_shape(variance)
                num_extra_dims = len(shape1) - len(shape2)
                for _ in range(num_extra_dims):
                    variance = K.expand_dims(variance, axis=1)
                hat_x_i_k = v_i_k / K.sqrt(variance + self.epsilon)
    
                hat_x_i = (tau_k / K.sqrt(prior_k + self.epsilon)) * hat_x_i_k
    
                normalized_x += hat_x_i
    
                updated_mean = K.mean(hat_x_i, axis=list(range(K.ndim(x)))[:-1])
                updated_var = K.var(hat_x_i, axis=list(range(K.ndim(x)))[:-1])
                updated_prior = K.mean(tau_k)
    
                self.mean[k, :].assign(self.momentum * self.mean[k, :] + (1.0 - self.momentum) * updated_mean)
                self.variance[k, :].assign(self.momentum * self.variance[k, :] + (1.0 - self.momentum) * updated_var)
                self.prior[k, 0].assign(self.momentum * self.prior[k, 0] + (1.0 - self.momentum) * updated_prior)
                
            else:
                hat_x_i_k = (x - mean_k) / (K.sqrt(var_k + self.epsilon))
                hat_x_i = (tau_k / K.sqrt(prior_k + self.epsilon)) * hat_x_i_k
                normalized_x += hat_x_i
                
        return normalized_x

