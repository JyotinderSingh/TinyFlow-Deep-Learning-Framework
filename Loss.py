import numpy as np


# Common loss class for regularization
class Loss:

    # Regularization loss calculation
    def regularization_loss(self, layer):

        # 0 by default
        regularization_loss = 0

        # L1 regularization - weights
        # Only calculate when factor greater than 0
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * \
                np.sum(np.abs(layer.weights))

        # L2 regularization - weights
        # Only calculate when factor greater than 0
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * \
                np.sum(layer.weights * layer.weights)

        # L1 regularization - biases
        # Only calculate when factor greater than 0
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * \
                np.sum(np.abs(layer.biases))

        # L2 regularization - biases
        # Only calculate when factor greater than 0
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * \
                np.sum(layer.biases * layer.biases)

        return regularization_loss


# Cross-entropy loss
class Loss_CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        '''Loss_CategoricalCrossEntropy.forward (predicted_values, ground_truth)\n
            Returns the negative_log_likelihood for the correct class score.\n
            The loss returned is the mean loss over the batch.
        '''

        # Number of samples in a batch
        samples = y_pred.shape[0]

        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            y_pred = y_pred[range(samples), y_true]

        # Losses
        negative_log_likelihoods = -np.log(y_pred)

        # Mask values - only for one-hot encoded labels
        if len(y_true.shape) == 2:
            negative_log_likelihoods *= y_true

        # Overall loss
        data_loss = np.sum(negative_log_likelihoods) / samples

        return data_loss

    # Backward pass
    def backward(self, dvalues, y_true):
        '''Loss_CategoricalCrossEntropy.backward (upstream_gradient, labels)\n
        Calculates the backward pass for the current loss function\n
        ---IMPLEMENTATION TO BE UPDATED SOON---'''

        samples = dvalues.shape[0]

        # Make a backup so we can safely modify
        self.dvalues = dvalues.copy()
        self.dvalues[range(samples), y_true] -= 1
        self.dvalues = self.dvalues / samples
