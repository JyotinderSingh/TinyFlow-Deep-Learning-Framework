import numpy as np

np.random.seed(0)


class Layer_Dense:
    '''Layer_Dense (input_size, neurons)'''

    def __init__(self, inputs, neurons):
        # Initialize weights and biases
        # we're doing (inputs, neurons) rather than (neurons, inputs) so that we don't have to perform a transpose everytime
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros(shape=(1, neurons))

    # Forward Pass
    def forward(self, inputs):
        '''Layer_Dense.forward (input_data)'''

        # Calculate the output values from inputs, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases
