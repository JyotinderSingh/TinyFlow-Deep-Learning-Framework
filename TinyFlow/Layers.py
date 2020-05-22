import numpy as np

np.random.seed(0)


class Layer_Dense:
    '''Layer_Dense (input_size, neurons)'''

    def __init__(self, inputs, neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regulariser_l1=0, bias_regulariser_l2=0):
        # Initialize weights and biases
        # we're doing (inputs, neurons) rather than (neurons, inputs) so that we don't have to perform a transpose everytime
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros(shape=(1, neurons))

        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regulariser_l1
        self.bias_regularizer_l2 = bias_regulariser_l2

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

        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = self.weights.copy()
            dL1[dL1 >= 0] = 1
            dL1[dL1 < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l1 * self.weights

        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = self.biases.copy()
            dL1[dL1 >= 0] = 1
            dL1[dL1 < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # Gradients on input values
        self.dvalues = np.dot(dvalues, self.weights.T)

    def __str__(self):
        return f"Dense: inputs: {self.weights.shape[0]}\tneurons: {self.weights.shape[1]}"


# Dropout
class Layer_Dropout:
    '''Layer_Dropout(rate)\n
    rate = amount of neurons you intend to disable
    '''

    # Init
    def __init__(self, rate):
        self.rate = 1 - rate

    # Forward pass
    def forward(self, values):
        # save input values
        self.input = values

        self.binary_mask = np.random.binomial(
            1, self.rate, size=values.shape) / self.rate

        # Apply mask to output values
        self.output = values * self.binary_mask

    # Backward pass
    def backward(self, dvalues):
        # Gradient on values
        self.dvalues = dvalues * self.binary_mask

    def __str__(self):
        return f"Dropout: rate: {self.rate}"