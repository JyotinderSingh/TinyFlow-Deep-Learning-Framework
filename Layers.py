import numpy as np

np.random.seed(0)


def create_data(points, classes, features=2):
    '''create_data (number_of_samples, no_of_classes)\n
        Each sample consists of 2 features (dimensions) by default
    '''

    # each generated sample will have 2 features (hardcoded for now)
    # data matrix (each row = single example)
    X = np.zeros((points*classes, features))
    y = np.zeros(points*classes, dtype='uint8')  # class labels
    for j in range(classes):
        ix = range(points*j, points*(j+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(j*4, (j+1)*4, points) + \
            np.random.randn(points)*0.2  # theta
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = j
    return X, y


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
