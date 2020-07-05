# Demo for Network wrapper to implement Multiclass Classification
import numpy as np
from TinyFlow.Network import Network
from TinyFlow.Datasets import spiral_data
from TinyFlow.Layers import Layer_Dense, Layer_Dropout
from TinyFlow.Activations import Activation_Softmax, Activation_ReLU
from TinyFlow.Optimizers import Optimizer_Adam
from TinyFlow.Loss import Loss_CategoricalCrossEntropy
from TinyFlow.Metrics import Accuracy_Categorical

# Create dataset
X, y = spiral_data(1000, 3)
X_test, y_test = spiral_data(100, 3)

# Instantiate the model
model = Network()

# Add layers
model.add(Layer_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regulariser_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(512, 3))
model.add(Activation_Softmax())

# Set loss, optimizer, and accuracy objects
model.set(loss=Loss_CategoricalCrossEntropy(), optimizer=Optimizer_Adam(
    learning_rate=0.05, decay=1e-8), accuracy=Accuracy_Categorical())

# Compile the model
model.compile_model()

# Train the model
model.train(X, y, epochs=10000, print_every=100, validation_data=(X_test, y_test))
