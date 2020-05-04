import numpy as np


# Cross-entropy loss
class Loss_CategoricalCrossEntropy:

    def forward(self, y_pred, y_true):
        '''Loss_CategoricalCrossEntropy.forward (predicted_values, ground_truth)\n
            Returns the negative_log_likelihood for the correct class score.\n
            The loss returned is the mean loss over the batch.
        '''

        # Number of samples in a batch
        samples = len(y_pred)

        # Probabilities for target values
        y_pred = y_pred[range(samples), y_true]

        # Losses
        negative_log_likelihoods = -np.log(y_pred)

        # Overall loss
        data_loss = np.mean(negative_log_likelihoods)

        return data_loss
