# C'est la bonne version j'ai modifié aussi le github
import tensorflow as tf

class UCBNorm(tf.keras.layers.Layer):
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
        min_value = -1.0
        max_value = 1.0
        return tf.random.uniform(shape=shape, minval=min_value, maxval=max_value, dtype=dtype)

    def custom_variance_initializer(self, shape, dtype=None):
        min_value = 0.001
        max_value = 0.01
        return tf.random.uniform(shape=shape, minval=min_value, maxval=max_value, dtype=dtype)

    def custom_prior_initializer(self, shape, dtype=None):
        initial_values = tf.random.uniform(shape, minval=0.01, maxval=0.99, dtype=dtype)
        normalized_values = initial_values / tf.reduce_sum(initial_values, axis=0)
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
            training = tf.keras.backend.learning_phase()
        x = inputs
        normalized_x = tf.zeros_like(x)
        mean = self.mean
        var = self.variance
        prior = self.prior
        num_expand_dims = len(x.shape) - 2

        for _ in range(num_expand_dims):
            mean = tf.expand_dims(mean, axis=1)
            var = tf.expand_dims(var, axis=1)

        var = tf.math.softplus(var)
        prior = tf.math.softmax(prior)

        for k in range(self.num_components):
            mean_k = mean[k, :]
            var_k = var[k, :]
            prior_k = prior[k, :]

            p_x_given_k = prior_k * tf.math.exp(-0.5 * ((x - mean_k) * (1 / (var_k + self.epsilon)) * (x - mean_k)))

            p_x_given_i = tf.zeros_like(p_x_given_k)
            for i in range(self.num_components):
                mean_i = mean[i, :]
                var_i = var[i, :]

                posterior_proba = prior[i] * tf.math.exp(
                    -0.5 * ((x - mean_i) * (1 / (var_i + self.epsilon)) * (x - mean_i)))

                p_x_given_i += posterior_proba

            tau_k = p_x_given_k / (p_x_given_i + self.epsilon)

            if training:
                sum_tau_k = tf.reduce_sum(tau_k, axis=list(range(x.ndim))[:-1])
                hat_tau_k = tau_k / (sum_tau_k + self.epsilon)

                prod = hat_tau_k * x
                expectation = tf.reduce_mean(prod, axis=list(range(x.ndim))[1:-1])
                shape1 = x.shape
                shape2 = expectation.shape
                num_extra_dims = len(shape1) - len(shape2)
                for _ in range(num_extra_dims):
                    expectation = tf.expand_dims(expectation, axis=1)
                v_i_k = x - expectation

                prod_bis = hat_tau_k * (prod * prod)
                variance = tf.reduce_mean(prod_bis, axis=list(range(x.ndim))[1:-1])
                shape1 = v_i_k.shape
                shape2 = variance.shape
                num_extra_dims = len(shape1) - len(shape2)
                for _ in range(num_extra_dims):
                    variance = tf.expand_dims(variance, axis=1)
                hat_x_i_k = v_i_k / tf.sqrt(variance + self.epsilon)

                hat_x_i = (tau_k / tf.math.sqrt(prior_k + self.epsilon)) * hat_x_i_k

                normalized_x += hat_x_i

                updated_mean = tf.reduce_mean(hat_x_i, axis=list(range(x.ndim))[:-1])
                updated_var = tf.math.reduce_variance(hat_x_i, axis=list(range(x.ndim))[:-1])
                updated_prior = tf.reduce_mean(tau_k)

                self.mean[k, :].assign(self.momentum * self.mean[k, :] + (1.0 - self.momentum) * updated_mean)
                self.variance[k, :].assign(self.momentum * self.variance[k, :] + (1.0 - self.momentum) * updated_var)
                self.prior[k, 0].assign(self.momentum * self.prior[k, 0] + (1.0 - self.momentum) * updated_prior)

            else:
                hat_x_i_k = (x - mean_k) / (tf.math.sqrt(var_k + self.epsilon))
                hat_x_i = (tau_k / tf.math.sqrt(prior_k + self.epsilon)) * hat_x_i_k
                normalized_x += hat_x_i

        return normalized_x

