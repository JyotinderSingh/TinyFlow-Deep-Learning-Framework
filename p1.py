import numpy as np

np.random.seed(0)


def create_data(points, classes):
    X = np.zeros((points*classes, 2))   # each generated sample will have 2 features (hardcoded for now)
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4,
                        points) + np.random.randn(points)*0.05
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y


class Layer_Dense:
    def __init__(self, inputs, neurons):
        # Initialize weights and biases
        # we're doing (inputs, neurons) rather than (neurons, inputs) so that we don't have to perform a transpose everytime
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros(shape=(1, neurons))

    # Forward Pass
    def forward(self, inputs):
        # Calculate the output values from inputs, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases


# Create Dataset
X, y = create_data(100, 3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)  # first dense layer, 2 inputs (each sample has 2 featues), 3 outputs

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Outputs of the first few samples
print(dense1.output[:5])







