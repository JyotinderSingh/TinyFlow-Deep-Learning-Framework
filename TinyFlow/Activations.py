import numpy as np


# ReLU activation
class Activation_ReLU:
    '''ReLU activation\n
    max (0, input)
    '''

    def forward(self, inputs):
        '''Activation_ReLU.forward (input_data)'''
        # np.maximum takes two inputs and finds element wise maximum
        self.output = np.maximum(0, inputs)

        # Cache the input values for backprop
        self.inputs = inputs

    # Backward pass
    def backward(self, dvalues):
        '''Activation_ReLU.backward (upstream_gradient)\n
        Calculates the backward pass through the current non linearity'''
        # Since we need to modify the original variable,
        # let's make a copy of the values first
        self.dvalues = dvalues.copy()

        # Zero gradient where the input values were <= 0
        self.dvalues[self.inputs <= 0] = 0

    # Calculate prediction for outputs
    def predictions(self, outputs):
        return outputs

    def __str__(self):
        return "ReLU Activation"


# Softmax activation
class Activation_Softmax:
    '''Softmax activation'''

    # Forward Pass
    def forward(self, inputs):
        '''Activation_Softmax.forward (input_data)'''

        # get unnormalized probabilities
        # np.max takes one input and finds maximum from that,
        # here it takes each row of the input (i.e. neuron activations for each sample)
        # and finds max in each of the rows
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        '''Activation_ReLU.backward (upstream_gradient)\n
        Calculates the backward pass through the current non linearity\n
        ---- IMPLEMENTATION TO BE UPDATED SOON---'''

        self.dvalues = dvalues.copy()

    # return predictions for outputs
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

    def __str__(self):
        return "Softmax Activation"


# Sigmoid activation
class Activation_Sigmoid:
    # Forward Pass
    def forward(self, inputs):
        # Save input and calcuilate/save output of the sigmoid function
        self.input = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    # Backward pass
    def backward(self, dvalues):
        # d/dx(sigm(x)) = sigm(x) * [1 - sigm(x)]
        self.dvalues = dvalues * (1 - self.output) * self.output

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return (outputs > 0.5) * 1

    def __str__(self):
        return "Sigmoid Activation"


# Linear activation
class Activation_Linear:

    # Forward pass
    def forward(self, inputs):
        # All you need to do is just cache the values
        self.input = inputs
        self.output = inputs

    # Backward pass
    def backward(self, dvalues):
        # Chain rule:
        # local Derivative = 1, upstream gradient = dvalues
        # downstream gradient = (local derivative ) * (upstream gradient)
        # self.dvalues = 1 * dvalues
        self.dvalues = dvalues.copy()

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs

    def __str__(self):
        return "Linear Activation"