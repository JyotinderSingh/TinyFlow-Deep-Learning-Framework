import numpy as np


# Common Accuracy class
class Accuracy:

    # Calculates accuracy, given predictions and ground truth values
    def calculate(self, predictions, y):

        # Get comparison results
        comparisons = self.compare(predictions, y)

        # Calculate an accuracy
        accuracy = np.mean(comparisons)

        # Return the accuracy
        return accuracy


class Accuracy_Regression(Accuracy):

    def __init__(self):
        # Create a precision property
        self.precision = None

    # Calculates precision value based on passed in ground truth values
    def init(self, y, reinit=False):
        '''init (self, y, reinit=False)\n
        reinit - Forces reinitilization of precision\n
        Calculates precision value based on passed in ground truth values
        '''

        if self.precision is None or reinit:
            self.precision = np.std(y) / 500

    # Compare predictions to the ground truth values
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision


class Accuracy_Categorical(Accuracy):

    # No initalization is needed
    def init(self, y):
        pass

    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        return predictions == y


# class Accuracy_BinaryLogisticRegression(Accuracy):

#     # No initialization is needed
#     def init(self, y):
#         pass

#     def compare(self, predictions, y):
#         return predictions == y


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
