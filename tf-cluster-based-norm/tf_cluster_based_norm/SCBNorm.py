import tensorflow as tf

class SCBNorm(tf.keras.layers.Layer):
    def __init__(self, num_clusters, epsilon=1e-3, **kwargs):
        """
        Initialize the Supervised Cluster-Based Normalization layer.

        Parameters:
        :param num_clusters: The number of clusters (prior knowledge)
        :param epsilon: A small positive value to prevent division by zero during normalization.
        """
        self.num_clusters = num_clusters
        self.epsilon = epsilon
        super(SCBNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Build the layer by creating sub-layers for learning initial mean and standard deviation.

        :param input_shape: The shape of the layer's input.

        This method initializes the layers for learning initial mean and standard deviation, based on the input shape.
        """
        self.input_dim = input_shape[0][-1]

        # Create weights for initial mean and standard deviation
        self.initial_mean = self.add_weight(
            name='initial_mean',
            shape=(self.num_clusters, self.input_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.initial_std = self.add_weight(
            name='initial_std',
            shape=(self.num_clusters, self.input_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        super(SCBNorm, self).build(input_shape)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
        	"num_clusters": self.num_clusters,
        	"epsilon": self.epsilon
        })
        return config

    def call(self, inputs):
        """
        Apply the Supervised Cluster-Based Normalization to the input data.

        :param inputs: A tuple of (x, cluster_id) where x is the data to be normalized, and cluster_id is the cluster identifier. Cluster identifier must be int32 format.


        :return normalized_x: The normalized output data.
        """
        x, cluster_id = inputs

        # Extract cluster indices from cluster_id
        indices = cluster_id[:, 0]

        # Gather initial mean and standard deviation based on cluster indices
        mean = tf.gather(self.initial_mean, indices)
        std = tf.gather(self.initial_std, indices)

        # Ensure standard deviation is positive
        std = tf.exp(std)

        # Determine the number of dimensions to expand
        num_expand_dims = len(x.shape) - 2

        # Expand mean and std dimensions accordingly
        for _ in range(num_expand_dims):
            mean = tf.expand_dims(mean, axis=1)
            std = tf.expand_dims(std, axis=1)

        # Perform normalization
        normalized_x = (x - mean) / (std + self.epsilon)

        return normalized_x
