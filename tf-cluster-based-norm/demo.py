import tensorflow as tf
from tf_cluster_based_norm import SCBNormBase, SCBNorm, UCBNorm


#### Create data
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

# Create target (5 classes)
labels = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]  
Y = tf.constant(labels)

# Create cluster (3 clusters)
cluster_indices = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0] 
cluster = tf.one_hot(cluster_indices, depth=3)




################################# Model with SCBNormBase layer ##############################

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
history = model.fit([X, cluster], Y, epochs=10)




######################### Model with SCBNorm layer   ############################################

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
history = model.fit([X, cluster], Y, epochs=10)




################### Model with UCBNorm layer (don't need prior knowledge)  #######################

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