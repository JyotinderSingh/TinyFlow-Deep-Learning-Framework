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

        # Cache the input values for backprop
        self.inputs = inputs

    # Backward pass
    def backward(self, dvalues):
        '''Layer_Dense.backward (upstream_gradient)\n
        Calculates the backward pass through the current layer
        '''

        # Gradiesnts on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradients on input values
        self.dvalues = np.dot(dvalues, self.weights.T)

    def __str__(self):
        return f"Dense: inputs: {self.weights.shape[0]}\tneurons: {self.weights.shape[1]}"