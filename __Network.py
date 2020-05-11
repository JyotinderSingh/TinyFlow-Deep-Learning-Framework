import numpy as np
import Layers
import Activations
import Loss
import Model
import Optimizers


class Network:
    def __init__(self, inputFeatures):
        self.layers = []
        self.inputFeatures = inputFeatures
        self.prev = -1
    
    def forwardPass(self):
        pass

    def backwardPass(self):
        pass

    def train(self, epochs):
        pass

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
