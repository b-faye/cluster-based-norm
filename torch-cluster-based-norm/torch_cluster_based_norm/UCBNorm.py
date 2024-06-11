# C'est la bonne version j'ai aussi modifié sur git
import torch
import torch.nn as nn
import torch.nn.functional as F


class UCBNorm(nn.Module):
    def __init__(self, input_dim, num_clusters, epsilon=1e-3, momentum=0.9):
        """
        Initialize the Unsupervised Cluster-Based Normalization layer.

        :param input_dim: The dimension of the layer's input.
        :param num_clusters: The number of clusters for normalization.
        :param epsilon: A small positive value to prevent division by zero during normalization.
        :param momentum: The momentum for updating mean, variance, and prior during training.
        """
        super(UCBNorm, self).__init__()
        self.input_dim = input_dim
        self.num_clusters = num_clusters
        self.epsilon = epsilon
        self.momentum = momentum

        self.mean = nn.Parameter(torch.Tensor(self.num_clusters, self.input_dim))
        self.variance = nn.Parameter(torch.Tensor(self.num_clusters, self.input_dim))
        self.prior = nn.Parameter(torch.Tensor(self.num_clusters, 1))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset parameters for mean, variance, and prior.
        """
        nn.init.uniform_(self.mean, -1.0, 1.0)
        nn.init.uniform_(self.variance, 0.001, 0.01)
        nn.init.uniform_(self.prior, 0.01, 0.99)
        self.prior.data /= torch.sum(self.prior.data, dim=0)

    def forward(self, x):
        """
        Apply the Unsupervised Cluster-Based Normalization to the input data.

        :param x: The input data to be normalized.

        :return normalized_x: The normalized output data.
        """
        normalized_x = torch.zeros_like(x)
        mean = self.mean
        var = self.variance
        prior = self.prior
        num_expand_dims = len(x.shape) - 2

        for _ in range(num_expand_dims):
            mean = mean.unsqueeze(1)
            var = var.unsqueeze(1)

        var = F.softplus(var)
        prior = F.softmax(prior, dim=0)

        for k in range(self.num_clusters):
            mean_k = mean[k, :]
            var_k = var[k, :]
            prior_k = prior[k, :]

            p_x_given_k = prior_k * torch.exp(-0.5 * ((x - mean_k) * (1 / (var_k + self.epsilon)) * (x - mean_k)))

            p_x_given_i = torch.zeros_like(p_x_given_k)
            for i in range(self.num_clusters):
                mean_i = mean[i, :]
                var_i = var[i, :]

                posterior_proba = prior[i] * torch.exp(-0.5 * ((x - mean_i) * (1 / (var_i + self.epsilon)) * (x - mean_i)))

                p_x_given_i += posterior_proba

            tau_k = p_x_given_k / (p_x_given_i + self.epsilon)
            
            if self.training:
                sum_tau_k = torch.sum(tau_k, dim=tuple(range(x.dim()))[:-1])
                hat_tau_k = tau_k / (sum_tau_k + self.epsilon)

                prod = hat_tau_k * x
                expectation = torch.mean(prod, dim=tuple(range(x.dim()))[1:-1])
                expectation = expectation.unsqueeze(-1).expand_as(x)

                v_i_k = x - expectation

                prod_bis = hat_tau_k * (prod * prod)
                variance = torch.mean(prod_bis, dim=tuple(range(x.dim()))[1:-1])
                variance = variance.unsqueeze(-1).expand_as(v_i_k)

                hat_x_i_k = v_i_k / torch.sqrt(variance + self.epsilon)

                hat_x_i = (tau_k / torch.sqrt(prior_k + self.epsilon)) * hat_x_i_k

                normalized_x += hat_x_i

                updated_mean = torch.mean(hat_x_i, dim=tuple(range(x.dim()))[:-1])
                updated_var = torch.var(hat_x_i, dim=tuple(range(x.dim()))[:-1])
                updated_prior = torch.mean(tau_k)

                self.mean[k, :].data = self.momentum * self.mean[k, :] + (1.0 - self.momentum) * updated_mean
                self.variance[k, :].data = self.momentum * self.variance[k, :] + (1.0 - self.momentum) * updated_var
                self.prior[k, 0].data = self.momentum * self.prior[k, 0] + (1.0 - self.momentum) * updated_prior

            else:
                hat_x_i_k = (x - mean_k) / (torch.sqrt(var_k + self.epsilon))
                hat_x_i = (tau_k / torch.sqrt(prior_k + self.epsilon)) * hat_x_i_k
                normalized_x += hat_x_i

        return normalized_x

    def call(self, x):
        
        return self.forward(x)
