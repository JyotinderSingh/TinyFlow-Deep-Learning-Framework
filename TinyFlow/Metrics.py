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
