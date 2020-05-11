import numpy as np


class Optimizer_SGD:
    '''Stochastic Gradient Descent Optimizer\n
    __init__(learning_rate, decay)\n
    Default learning rate is 1.0, default decay is 0.1
    '''

    # Initialize optimizer - set settings,
    # learning rate of 1 is default for this optimizer
    def __init__(self, learning_rate=1.0, decay=0.1):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    # Call once before any parameter updates
    def pre_update_params(self):
        '''Optimizer_SGD.pre_update_params()\n
        Called before parameter update to decay the learning rate
        '''

        # If we have a decay other than 0, we update learning rate
        if self.decay:
            self.current_learning_rate = self.current_learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        '''Optimizer_SGD.update_params (layer)\n
        Update the parameters to move the loss closer to the global minima
        '''

        layer.weights -= self.current_learning_rate * layer.dweights
        layer.biases -= self.current_learning_rate * layer.dbiases

    def post_update_params(self):
        '''Optimizer_SGD.post_update_params()\n
        Called after parameter update to increase iteration count for decay
        '''

        self.iterations += 1
