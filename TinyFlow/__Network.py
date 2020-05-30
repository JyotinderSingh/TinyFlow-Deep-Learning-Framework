import numpy as np
from TinyFlow import Layers
from TinyFlow import Activations
from TinyFlow import Loss
from TinyFlow import Model
from TinyFlow import Optimizers


class Network:
    '''Network(inputFeatures)
    Creates a new Neural Network object.
    inputFeatures: Dimensions per sample
    '''


    def __init__(self, inputFeatures):
        self.layers = []
        self.inputFeatures = inputFeatures
        self.prev = -1

    def forwardPass(self):
        # will be used for refactoring later
        pass

    def backwardPass(self):
        # will be used for refactoring later
        pass

    def train(self, input, labels, epochs, lossFunction, optimizer):
        '''Performs a forward pass on the input data through the network for the
        specified number of epochs.
        parameters:
        input: np.array object\n
        labels: np.array object\n
        epochs: integer\n
        lossFunction: instance of some loss function (eg. Loss_CategoricalCrossEntropy)\n
        optimizer: instance of an optimizer (eg. Adam, AdaGrad, SGD etc.)
        '''

        assert input.shape[1] == self.inputFeatures
        assert len(self.layers) > 0

        for epoch in range(epochs):
            inputLayerOutput = self.layers[0].forward(input)


            # Forward pass
            for idx in range(1, len(self.layers)):
                self.layers[idx].forward(self.layers[idx - 1].output)

            # Get metrics
            loss = lossFunction.forward(self.layers[-1].output, labels)
            accuracy = Model.model_accuracy(self.layers[-1].output, labels)

            if not epoch % 100:
                print(
                    f'\nepoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {optimizer.current_learning_rate}')

            # Backward pass
            lossFunction.backward(self.layers[-1].output, labels)
            self.layers[-1].backward(lossFunction.dvalues)
            for idx in range(len(self.layers) - 2, -1, -1):
                self.layers[idx].backward(self.layers[idx + 1].dvalues)

            # Update weights
            optimizer.pre_update_params()
            for idx in range(len(self.layers)):
                if isinstance(self.layers[idx], Layers.Layer_Dense):
                    optimizer.update_params(self.layers[idx])            
            optimizer.post_update_params


    def addDenseLayer(self, neurons):
        if len(self.layers) == 0:
            denseX = Layers.Layer_Dense(self.inputFeatures, neurons)
            self.layers.append(denseX)
        else:
            denseX = Layers.Layer_Dense(
                self.layers[self.prev].weights.shape[1], neurons)
            self.layers.append(denseX)
        self.prev = len(self.layers) - 1

    def addReLU(self):
        reluX = Activations.Activation_ReLU()
        self.layers.append(reluX)

    def addSoftmax(self):
        softmaxX = Activations.Activation_Softmax()
        self.layers.append(softmaxX)

    def getSummary(self):
        summary = ""
        for i in range(len(self.layers)):
            summary += f"Layer {str(i)} <" + self.layers[i].__str__() + ">\n"
        summary = summary.strip()
        return summary