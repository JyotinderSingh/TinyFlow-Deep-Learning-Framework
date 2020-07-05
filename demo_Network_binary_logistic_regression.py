# Demo for Network wrapper to implement Binary Logistic Regression
import numpy as np
from TinyFlow.Network import Network
from TinyFlow.Datasets import spiral_data
from TinyFlow.Layers import Layer_Dense
from TinyFlow.Activations import Activation_Sigmoid, Activation_ReLU
from TinyFlow.Optimizers import Optimizer_Adam
from TinyFlow.Loss import Loss_BinaryCrossEntropy
from TinyFlow.Metrics import Accuracy_Categorical

# Create train and test set
X, y = spiral_data(100, 2)
X_test, y_test = spiral_data(100, 2)

# Reshape labels to be a list of lists
# Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
y = y.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Instantiate the model
model = Network()

# Add layers
model.add(Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regulariser_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 1))
model.add(Activation_Sigmoid())

# Set loss, optimizer, and accuracy
model.set(loss=Loss_BinaryCrossEntropy(), optimizer=Optimizer_Adam(
    decay=1e-8), accuracy=Accuracy_Categorical())

# Compile the model
model.compile_model()

# Train the model
model.train(X, y, epochs=10000, print_every=100,
            validation_data=(X_test, y_test))
