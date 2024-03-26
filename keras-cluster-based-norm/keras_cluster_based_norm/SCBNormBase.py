import keras
from keras import layers, utils

@utils.register_keras_serializable()
class SCBNormBase(layers.Layer):
    def __init__(self, epsilon=1e-3, **kwargs):
        """
        Initialize the Supervised Cluster-Based Normalization Base layer a Tiny Version Supervised Cluster-Based Normalization.

        :param epsilon: A small positive value to prevent division by zero during normalization.
        """
        super(SCBNormBase, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        """
        Build the layer by creating sub-layers for learning mean and standard deviation.

        Parameters:
        :param input_shape: The shape of the layer's input.

        This method initializes the layers for learning mean and standard deviation, based on the input shape.
        """
        self.input_dim = input_shape[0][-1]
        self.cluster_dim = input_shape[1][-1]

        # Layer for learning mean
        self.mean_layer = layers.Dense(
            units=self.input_dim,
            activation=None,
            kernel_initializer='glorot_uniform',
            trainable=True,
            input_shape=(self.cluster_dim,)
        )

        # Layer for learning standard deviation
        self.std_layer = layers.Dense(
            units=self.input_dim,
            activation=None,
            kernel_initializer='glorot_uniform',
            trainable=True,
            input_shape=(self.cluster_dim,)
        )

        super(SCBNormBase, self).build(input_shape)
        
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
        	"epsilon": self.epsilon
        })
        return config
        
        
    def call(self, inputs):
        """
        Apply the Tiny Supervised Cluster-Based Normalization to the input data.

        :param inputs: A tuple consisting of two elements: 'x', which represents the data to be normalized, and 'cluster_id', which serves as the cluster identifier. The 'cluster_id' is encoded in one-hot format with int32 values..

        
        :return normalized_x: The normalized output data.
        """
        x, cluster_id = inputs

        # Calculate mean and standard deviation from cluster_id
        mean = self.mean_layer(cluster_id)
        std = self.std_layer(cluster_id)

        # Ensure standard deviation is positive
        std = keras.backend.exp(std)

        # Determine the number of dimensions to expand
        num_expand_dims = len(x.shape) - 2

        # Expand mean and std dimensions accordingly
        for _ in range(num_expand_dims):
            mean = keras.backend.expand_dims(mean, axis=1)
            std = keras.backend.expand_dims(std, axis=1)

        # Perform normalization
        normalized_x = (x - mean) / (std + self.epsilon)

        return normalized_x
