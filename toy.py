######################################
########## TESTING BACKPROP ##########
######################################
import numpy as np

# now we have 3 samples (feature sets) of data
inputs = np.array([[1, 2, 3, 2.5],
                  [2., 5., -1., 2],
                  [-1.5, 2.7, 3.3, -0.8]])

# now we have 3 sets of weights - one for each neuron
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

# one bias for each neuoron
biases = np.array([[2], [3], [0.5]]).T

# Forward pass

# Forward pass through dense layer
layer_outputs = np.dot(inputs, weights) + biases

# Foward pass through ReLU activation
relu_outputs = np.maximum(0, layer_outputs)

# Let's optimize and test backprop here
# ReLU activation
relu_dvalues = np.ones(relu_outputs.shape)  # simulates derivative
relu_dvalues[layer_outputs <= 0] = 0
drelu = relu_dvalues

# Dense layer
dinputs = np.dot(drelu, weights.T)  # dinputs - multiply by weights
dweights = np.dot(inputs.T, drelu)   # dweights multiply by inputs

# sbiases - sum values, do this over samples (first axis), keepdims
# as this by default will produce a plain list
dbiases = np.sum(drelu, axis=0, keepdims=True)

# Update parameters
weights += -0.001 * dweights
biases += -0.001 * dbiases

print(weights)
print(biases)
