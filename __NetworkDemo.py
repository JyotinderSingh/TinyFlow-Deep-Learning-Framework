# ------------------------------#
# __Network.py is still under development
# expect weird bugs in weird scenarios
# ------------------------------#

from __Network import Network
from Loss import Loss_CategoricalCrossEntropy
from Optimizers import Optimizer_Adam
from Model import create_data

# Create example data (each sample has 2 dimensions by default)
# number of samples: 100, number of classes: 3
X, y = create_data(100, 3)

# Create a Network object, and pass 2 to it as each of our
# input sample has 2 features
net = Network(2)
net.addDenseLayer(64)   # Add FC layer with 64 neurons
net.addReLU()   # ReLU activation
net.addDenseLayer(128)
net.addReLU()
net.addDenseLayer(3)
net.addSoftmax()    # Softmax activation for final results
print(net.getSummary()) # prints a sumary of the network architecture

loss_function = Loss_CategoricalCrossEntropy()  # Instantiate a loss function
optimizer = Optimizer_Adam(learning_rate=0.05, decay=4e-8)  # Instantiate an optimizer

# Run the training loop
net.train(X, y, 10001, loss_function, optimizer)
