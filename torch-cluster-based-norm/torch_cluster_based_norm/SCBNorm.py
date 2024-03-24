import torch
import torch.nn as nn
import torch.nn.functional as F

class SCBNorm(nn.Module):
    def __init__(self, num_clusters, input_dim, epsilon=1e-3):
        """
        Initialize the Supervised Cluster-Based Normalization layer.

        Parameters:
        :param num_clusters: The number of clusters (prior knowledge)
        :param epsilon: A small positive value to prevent division by zero during normalization.
        """
        super(SCBNorm, self).__init__()
        self.num_clusters = num_clusters
        self.epsilon = epsilon
        self.input_dim = input_dim

        # Define initial mean and standard deviation as learnable parameters
        self.initial_mean = nn.Parameter(torch.Tensor(num_clusters, self.input_dim))
        self.initial_std = nn.Parameter(torch.Tensor(num_clusters, self.input_dim))

        # Initialize parameters
        nn.init.xavier_uniform_(self.initial_mean)
        nn.init.xavier_uniform_(self.initial_std)

    def forward(self, inputs):
        """
        Apply the Supervised Cluster-Based Normalization to the input data.

        :param inputs: A tuple of (x, cluster_id) where x is the data to be normalized, and cluster_id is the cluster identifier. Cluster identifier must be int32 format.

        :return normalized_x: The normalized output data.
        """
        x, cluster_id = inputs

        # Extract cluster indices from cluster_id
        indices = cluster_id

        # Gather initial mean and standard deviation based on cluster indices
        mean = self.initial_mean[indices]
        std = self.initial_std[indices]

        # Ensure standard deviation is positive
        std = torch.exp(std)

        # Determine the number of dimensions to expand
        num_expand_dims = x.dim() - 2

        # Expand mean and std dimensions accordingly
        for _ in range(num_expand_dims):
            mean = mean.unsqueeze(1)
            std = std.unsqueeze(1)

        # Perform normalization
        normalized_x = (x - mean) / (std + self.epsilon)

        return normalized_x

    def call(self, inputs):
        return self.forward(inputs)