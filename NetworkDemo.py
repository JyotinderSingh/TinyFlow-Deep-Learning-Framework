from TinyFlow.Network import Network
from TinyFlow.Loss import Loss_CategoricalCrossEntropy
from TinyFlow.Optimizers import Optimizer_Adam
from TinyFlow.Model import create_data

# Create example data (each sample has 2 dimensions by default)
# number of samples: 1000, number of classes: 3
X, y = create_data(1000, 3)

# Create a Network object, and pass 2 to it as each of our
# input sample has 2 features
net = Network(2)
# Add FC layer with 64 neurons
net.addDenseLayer(64, weight_regularizer_l2=1e-5, bias_regulariser_l2=1e-5)
net.addReLU()   # ReLU activation
net.addDropoutLayer(0.1)
net.addDenseLayer(128)
net.addReLU()
net.addDenseLayer(3)
net.addSoftmax()    # Softmax activation for final results
print(net.getSummary())  # prints a sumary of the network architecture

loss_function = Loss_CategoricalCrossEntropy()  # Instantiate a loss function
# Instantiate an optimizer
optimizer = Optimizer_Adam(learning_rate=0.05, decay=4e-8)

# Run the training loop
net.train(X, y, 10001, loss_function, optimizer)

# Test the accuracy on unseen data
X_test, y_test = create_data(100, 3)

net.test(X_test, y_test, loss_function)
