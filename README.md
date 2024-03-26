# Cluster-Based Normalization




Cluster-Based Normalization (CB-Norm) is a novel normalization technique introduced to address challenges encountered during neural network training. It leverages a Gaussian mixture model to specifically target issues related to gradient stability and learning acceleration. CB-Norm introduces a one-step normalization approach where parameters of each mixture component serve as weights for deep neural networks. We propose Three versions of CB-Norm:

- **Supervised Cluster-Based Normalization (SCB-Norm):**


- **Supervised Cluster-Based Normalization (SCB-Norm-Base)**


- **Unsupervised Cluster-Based Normalization (UCB-Norm):**


## References


- **All versions:** *Cluster-Based Normalization Layer for Neural Networks*, FAYE et al., [ArXiv Link](https://arxiv.org/abs/2403.16798)


## Usage

### Tensorflow
- For manual installation, navigate to the directory named "tf-cluster-based-norm".
    ```bash
    git clone git@github.com:b-faye/cluster-based-norm.git
    cd tf-cluster-based-norm
    pip install dist/tf_cluster_based_norm-1.0.tar.gz
    ```

- For online installation, please follow the provided instructions:
    ```bash
    pip install tf-cluster-based-norm
    ```
    ```python
    from tf_cluster_based_norm import SCBNorm, SCBNormBase, UCBNorm
    scb_norm = SCBNorm(num_clusters=10)
    scb_norm_base = SCBNormBase()
    ucb_norm = UCBNorm(num_clusters=10)
    ```

### Keras
- For manual installation, navigate to the directory named "keras-cluster-based-norm".
    ```bash
    git clone git@github.com:b-faye/cluster-based-norm.git
    cd keras-cluster-based-norm
    pip install dist/keras_cluster_based_norm-1.0.tar.gz
    ```
- For online installation, please follow the provided instructions:
    ```bash
    pip install keras-cluster-based-norm
    ```
    ```python
    from keras_cluster_based_norm import SCBNorm, SCBNormBase, UCBNorm
    scb_norm = SCBNorm(num_clusters=10)
    scb_norm_base = SCBNormBase()
    ucb_norm = UCBNorm(num_clusters=10)
    ```

### PyTorch
- For manual installation, navigate to the directory named "torch-cluster-based-norm".
    ```bash
    git clone git@github.com:b-faye/cluster-based-norm.git
    cd torch-cluster-based-norm
    pip install dist/torch_cluster_based_norm-1.0.tar.gz
    ```

- For online installation, please follow the provided instructions:
    ```bash
    pip install torch-cluster-based-norm
    ```
    ```python
    from torch_cluster_based_norm import SCBNorm, SCBNormBase, UCBNorm
    scb_norm = SCBNorm(num_clusters=10, input_dim=5)
    scb_norm_base = SCBNormBase(num_clusters=10, input_dim=5)
    ucb_norm = UCBNorm(num_clusters=10, input_dim=5)
    ```
