import numpy as np

np.random.seed(0)


def create_data(points, classes, features=2):
    '''create_data (number_of_samples, no_of_classes)\n
        Each sample consists of 2 features (dimensions) by default
    '''

    # each generated sample will have 2 features (hardcoded for now)
    # data matrix (each row = single example)
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4,
                        points) + np.random.randn(points)*0.05
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y


def model_accuracy(outputs, labels):
    '''Returns the accuracy of the model on the current batch'''

    # Calculate the accuracy from output and targets
    predictions = np.argmax(outputs, axis=1)    # calculate values along first axis

    accuracy = np.mean(predictions == labels)

    return accuracy