import tensorflow as tf


class UCBNorm(tf.keras.layers.Layer):
    def __init__(self, num_clusters, epsilon=1e-3, momentum=0.9, **kwargs):
        """
        Initialize the Unsupervised Cluster-Based Normalization layer.

        :param num_clusters: The number of clusters for normalization.
        - epsilon: A small positive value to prevent division by zero during normalization.
        - momentum: The momentum for updating mean, variance, and prior during training.
        """
        self.num_clusters = num_clusters
        self.epsilon = epsilon
        self.momentum = momentum
        super(UCBNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Build the layer by initializing weights for mean, variance, and prior.


        :param input_shape: The shape of the layer's input.
        """
        self.input_dim = input_shape[-1]
        self.mean = self.add_weight(
            name='mean',
            shape=(self.num_clusters, self.input_dim),
            initializer=self.custom_mean_initializer,
            trainable=True
        )
        self.variance = self.add_weight(
            name='variance',
            shape=(self.num_clusters, self.input_dim),
            initializer=self.custom_variance_initializer,
            trainable=True
        )
        self.prior = self.add_weight(
            name='prior',
            shape=(self.num_clusters, 1),
            initializer=self.custom_prior_initializer,
            trainable=True
        )
        super(UCBNorm, self).build(input_shape)

    def custom_mean_initializer(self, shape, dtype=None):
        """
        Custom initializer for means.


        :param shape: The shape of the tensor to initialize.
        :param dtype: The data type of the tensor.

        :return A tensor with initialized mean values.
        """
        # Initialize means with random values
        min_value = -1.0  # Minimum value for means
        max_value = 1.0   # Maximum value for means
        return tf.random.uniform(shape=shape, minval=min_value, maxval=max_value, dtype=dtype)

    def custom_variance_initializer(self, shape, dtype=None):
        """
        Custom initializer for variances.

        :param shape: The shape of the tensor to initialize.
        :param dtype: The data type of the tensor.

        :param A tensor with initialized variance values.
        """
        # Initialize variances with strictly positive random values
        min_value = 0.001  # Minimum value for variance
        max_value = 0.01   # Maximum value for variance
        return tf.random.uniform(shape=shape, minval=min_value, maxval=max_value, dtype=dtype)

    def custom_prior_initializer(self, shape, dtype=None):
        """
        Custom initializer for priors.


        :param shape: The shape of the tensor to initialize.
        :param dtype: The data type of the tensor.


        :return A tensor with initialized prior values.
        """
        # Initialize priors with values between ]0, 1[ such that the sum is equal to 1
        initial_values = tf.random.uniform(shape, minval=0.01, maxval=0.99, dtype=dtype)
        normalized_values = initial_values / tf.reduce_sum(initial_values, axis=0)
        return normalized_values

    def get_config(self):
        """
        Get the configuration of the layer.


        :param config: A dictionary containing the layer configuration.
        """
        config = super(UCBNorm, self).get_config().copy()
        config.update(
            {
                "num_clusters": self.num_clusters,
                "epsilon": self.epsilon,
                "momentum": self.momentum
            })
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, training=None):
        """
        Apply the Unsupervised Cluster-Based Normalization to the input data.

        :param inputs: The input data to be normalized.
        :param training: A boolean indicating whether the layer is in training mode or not.


        :return normalized_x: The normalized output data.
        """
        if training is None:
            # 1: for training and 0 for inference
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

        for k in range(self.num_clusters):
            mean_k = mean[k, :]
            var_k = var[k, :]
            prior_k = prior[k, :]

            p_x_given_k = prior_k * tf.math.exp(-0.5 * ((x - mean_k) * (1 / (var_k + self.epsilon)) * (x - mean_k)))

            p_x_given_i = tf.zeros_like(p_x_given_k)
            for i in range(self.num_clusters):
                mean_i = mean[i, :]
                var_i = var[i, :]

                posterior_proba = prior[i] * tf.math.exp(-0.5 * ((x - mean_i) * (1 / (var_i + self.epsilon)) * (x - mean_i)))

                p_x_given_i += posterior_proba

            tau_k = p_x_given_k / (p_x_given_i + self.epsilon)

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

            if training:
                updated_mean = tf.reduce_mean(hat_x_i, axis=list(range(x.ndim))[:-1])
                updated_var = tf.math.reduce_variance(hat_x_i, axis=list(range(x.ndim))[:-1])
                updated_prior = tf.reduce_mean(tau_k)

                
                self.mean[k, :].assign(self.momentum * self.mean[k, :] + (1.0 - self.momentum) * updated_mean)
                self.variance[k, :].assign(self.momentum * self.variance[k, :] + (1.0 - self.momentum) * updated_var)
                self.prior[k, 0].assign(self.momentum * self.prior[k, 0] + (1.0 - self.momentum) * updated_prior)

        return normalized_x
