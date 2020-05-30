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

    def __str__(self):
        return "ReLU"


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

    def __str__(self):
        return "Softmax"