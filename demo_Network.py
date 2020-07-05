######################
## UNDER DEVELOPMENT##
######################

# Demo for Network wrapper
from TinyFlow.Network import Network
from TinyFlow.Datasets import sine_data
from TinyFlow.Layers import Layer_Dense
from TinyFlow.Activations import Activation_Linear, Activation_ReLU
from TinyFlow.Optimizers import Optimizer_Adam
from TinyFlow.Loss import Loss_MeanSquaredError

# Create a dataset
X, y = sine_data()

# instantiate the model
model = Network()

# Add layers
model.add(Layer_Dense(1, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 1))
model.add(Activation_Linear())

# Set loss and optimizer objects
model.set(loss=Loss_MeanSquaredError(), optimizer=Optimizer_Adam(decay=1e-8))

# Compile the model
model.compile_model()

# Train the model
model.train(X, y, epochs=10000, print_every=100)