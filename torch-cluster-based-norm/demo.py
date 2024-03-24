import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim



# Create data
data = np.array([[1, 2, 3, 4, 5],
                 [6, 7, 8, 9, 10],
                 [11, 12, 13, 14, 15],
                 [16, 17, 18, 19, 20],
                 [21, 22, 23, 24, 25],
                 [26, 27, 28, 29, 30],
                 [31, 32, 33, 34, 35],
                 [36, 37, 38, 39, 40],
                 [41, 42, 43, 44, 45],
                 [46, 47, 48, 49, 50]])

X = torch.tensor(data, dtype=torch.float32)

# Create target (5 classes)
labels = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]  
Y = torch.tensor(labels, dtype=torch.long)

# Create cluster (3 clusters)
cluster_indices = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0] 
cluster = torch.tensor(np.eye(3)[cluster_indices], dtype=torch.float32)



######################### Model with SCBNormBase layer   ############################################



# Apply normalization layer
scb_layer = SCBNormBase()

# Define the rest of your model architecture
# For example:
hidden_layer = nn.Linear(5, 10)
output_layer = nn.Linear(10, 10)

# Define the model
model = nn.Sequential(
    scb_layer,
    nn.ReLU(),
    hidden_layer,
    nn.ReLU(),
    output_layer
)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model([X, cluster])
    loss = criterion(output, Y)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')




######################### Model with SCBNorm layer   ############################################

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.hidden_layer = nn.Linear(5, 10)
        self.scb_norm = SCBNorm(num_clusters=3, input_dim=10)  
        self.output_layer = nn.Linear(10, 5)

    def forward(self, x, cluster_id):
        hidden_output = self.hidden_layer(x)
        normalized_activation = self.scb_norm((hidden_output, cluster_id))
        output = self.output_layer(normalized_activation)
        return output

# Instantiate the model
model = MyModel()

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Use cluster indices as prior knowledge
cluster = torch.tensor(cluster_indices)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X, cluster)
    loss = criterion(outputs, Y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss for monitoring training progress
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")



################### Model with UCBNorm layer (don't need prior knowledge)  #######################


# Define the model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.ucb_norm = UCBNorm(num_clusters=3, input_dim=5)
        self.hidden_layer = nn.Linear(5, 10)
        self.output_layer = nn.Linear(10, 5)

    def forward(self, x):
        x = self.ucb_norm(x)
        x = F.relu(self.hidden_layer(x))
        x = F.softmax(self.output_layer(x), dim=1)
        return x

# Instantiate the model
model = MyModel()

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()


# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, Y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss for monitoring training progress
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
