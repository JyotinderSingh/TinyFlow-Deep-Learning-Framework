import numpy as np

np.random.seed(0)


# Spiral Classification Dataset
def spiral_data(points, classes):
    '''spiral_data (number_of_samples, no_of_classes)\n
        Each sample consists of 2 features (dimensions) by \n
        Useful for generating classification data
    '''

    # each generated sample will have 2 features (hardcoded for now)
    # data matrix (each row = single example)
    # data matrix (each row = single example)
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')  # class labels
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4,
                        points) + np.random.randn(points)*0.2   # theta
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y


# Sine dataset
def sine_data(samples=10000):
    ''' sin_data (number_of_samples)\n
    Useful for generating regression data
    '''
    X = np.arange(samples).reshape(-1, 1) / samples
    y = np.sin(2 * np.pi * X).reshape(-1, 1)
    return X, y
