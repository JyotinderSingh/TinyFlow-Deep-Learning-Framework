import numpy as np


class Optimizer_SGD:
    '''Stochastic Gradient Descent Optimizer with Momentum\n
    __init__(learning_rate, decay)\n
    Default learning rate is 1.0, default decay is 0.1
    '''

    # Initialize optimizer - set settings,
    # learning rate of 1 is default for this optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

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
        '''Optimizer_SGD.update_params(layer)\n
        Update the parameters to move the loss closer to the global minima
        '''

        # If layer does not contain momentum arrays, create them
        # filled with zeros
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)

        # If we use momentum (i.e. momentum != 0)
        if self.momentum:

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update
            # with current gradients
            weight_updates = (
                (self.momentum * layer.weight_momentums) -
                (self.current_learning_rate * layer.dweights)
            )

            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = (
                (self.momentum * layer.bias_momentums) -
                (self.current_learning_rate * layer.dbiases)
            )

            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (without momentum)
        else:
            weight_updates = (-self.current_learning_rate * layer.dweights)
            bias_updates = (-self.current_learning_rate * layer.dbiases)

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        '''Optimizer_SGD.post_update_params()\n
        Called after parameter update to increase iteration count for decay
        '''

        self.iterations += 1
