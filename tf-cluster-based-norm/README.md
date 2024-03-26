# Cluster-Based Normalization with Tensorflow

Cluster-Based Normalization (CB-Norm) is a set of normalization layers designed for neural networks, accommodating both supervised and unsupervised learning scenarios. These layers aim to adaptively normalize data based on prior information, which can be beneficial in various machine learning tasks.

The Supervised versions (SCB-Norm and SCB-Norm-Base) are designed for scenarios where prior knowledge (clusters) is available and needs to be considered during normalization. These clusters, defined by experts or derived from clustering algorithms, can include classes, superclasses, or domains in domain adaptation scenarios.


## References


- **All versions:** *Cluster-Based Normalization Layer for Neural Networks*, FAYE et al., [ArXiv Link](https://arxiv.org/abs/2403.16798)


## Installation

To install the Cluster-Based Normalization package with **TensorFlow** via pip, use the following command::

```bash
pip install tf-cluster-based-norm
```

## Usage

### Generate Data

```python
import tensorflow as tf
from tf_cluster_based_norm import SCBNormBase

# Create data and cluster
data = [[1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
        [26, 27, 28, 29, 30],
        [31, 32, 33, 34, 35],
        [36, 37, 38, 39, 40],
        [41, 42, 43, 44, 45],
        [46, 47, 48, 49, 50]]

X = tf.constant(data)

# Establishing clusters (3 clusters): SCBNorm employs indices as input for normalizing, while SCBNormBase utilizes a one-hot representation of indices as input.
cluster_indices = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0] 
cluster_scb_norm = tf.constant(cluster_indices, shape=(10,1), dtype=tf.int32)
cluster_scb_norm_base = tf.one_hot(cluster_indices, depth=3)

```


### Tiny Supervised Cluster-Based Normalization (SCBNormBase)

SCBNormBase is a soft version of Supervised Cluster-Based Normalization. It also requires data and cluster as input, but it provides a lighter approach compared to Supervised Cluster-Based Normalization.

```python
import tensorflow as tf
from tf_cluster_based_norm import SCBNormBase

# Define input shapes
X_shape = (10, 5)  
cluster_shape = (10, 3)

# Define inputs
X_input = tf.keras.Input(shape=X_shape[1:])  
cluster_input = tf.keras.Input(shape=cluster_shape[1:], dtype=tf.int32) 

# Apply normalization layer
normalized_X = SCBNormBase()((X_input, cluster_input))

# Define the rest of your model architecture
# For example:
hidden_layer = tf.keras.layers.Dense(units=10, activation='relu')(normalized_X)
output_layer = tf.keras.layers.Dense(units=10, activation='softmax')(hidden_layer)

# Define the model
model = tf.keras.Model(inputs=[X_input, cluster_input], outputs=output_layer)

# Compile the model (you can specify your desired optimizer, loss, and metrics)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
history = model.fit([X, cluster_scb_norm_base], Y, epochs=10)

```

### Supervised Cluster-Based Normalization (SCBNorm)

SCBNorm adapts the normalization process based on prior information (cluster) provided alongside the input data. By incorporating such cluster, the layer enhances the normalization process, leading to accelerate convergence and improved performance.

```python
from tf_cluster_based_norm import SCBNorm

# Define input shapes
X_shape = (10, 5)  
cluster_shape = (10, 3)

# Define inputs
X_input = tf.keras.Input(shape=X_shape[1:])  
cluster_input = tf.keras.Input(shape=cluster_shape[1:], dtype=tf.int32) 

# Define the rest of your model architecture
# For example:
hidden_layer = tf.keras.layers.Dense(units=10, activation='relu')(X_input)

# Apply normalization layer
normalized_activation = SCBNorm(num_clusters=3)((hidden_layer, cluster_input))

output_layer = tf.keras.layers.Dense(units=10, activation='softmax')(normalized_activation)

# Define the model
model = tf.keras.Model(inputs=[X_input, cluster_input], outputs=output_layer)

# Compile the model (you can specify your desired optimizer, loss, and metrics)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Fit the model
history = model.fit([X, cluster_scb_norm], Y, epochs=10)

```

### Unsupervised Cluster-Based Normalization (UCB-Norm)

This version doesn't require explicit prior information and adapts based on the input data distribution.

```python
from tf_cluster_based_norm import UCBNorm

# Define input shapes
X_shape = (10, 5)  

# Define inputs
X_input = tf.keras.Input(shape=X_shape[1:])  

# Apply normalization layer
normalized_X = UCBNorm(num_clusters=3)(X_input)

# Define the rest of your model architecture
# For example:
hidden_layer = tf.keras.layers.Dense(units=10, activation='relu')(normalized_X)
output_layer = tf.keras.layers.Dense(units=10, activation='softmax')(hidden_layer)

# Define the model
model = tf.keras.Model(inputs=X_input, outputs=output_layer)

# Compile the model (you can specify your desired optimizer, loss, and metrics)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
history = model.fit(X, Y, epochs=10)

```


This README provides an overview of the Cluster-Based Normalization package along with examples demonstrating the usage of different normalization layers. You can modify and extend these examples according to your specific requirements.