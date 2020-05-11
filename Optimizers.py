import numpy as np


class Optimizer_SGD:
    '''Stochastic Gradient Descent Optimizer\n
    __init__(learning_rate)\n
    Default learning rate is 1.0
    '''

    # Initialize optimizer - set settings,
    # learning rate of 1 is default for this optimizer
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    # Update parameters
    def update_params(self, layer):
        '''Optimizer_SGD.update_params (layer)\n
        Update the parameters to move the loss closer to the global minima
        '''

        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases
