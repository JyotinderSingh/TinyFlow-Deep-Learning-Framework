import numpy as np


def model_accuracy_softmax(outputs, labels):
    '''Returns the accuracy of the model on the current batch'''

    # Calculate the accuracy from output and targets
    # calculate values along first axis
    predictions = np.argmax(outputs, axis=1)

    accuracy = np.mean(predictions == labels)

    return accuracy


def model_accuracy_sigmoid(output, labels):
    '''Returns the accuracy of the model on the current batch'''

    predictions = (output > 0.5) * 1

    accuracy = np.mean(predictions == labels)

    return accuracy


def model_accuracy_linear(output, labels, accuracy_precision):
    '''Returns the accuracy of the model on the current batch'''

    # Calculate accuracy from output of activation2 and targets
    # To calculate it we're taking absolute difference between
    # predictions and ground truth values and compare if differences
    # are lower than given precision value
    accuracy = np.mean(np.absolute(output - labels) < accuracy_precision)

    return accuracy
